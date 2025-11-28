"""
MiniMind Full SFT Training using MLX
Optimized for Apple Silicon (M1/M2/M3/M4)

Supervised Fine-Tuning on instruction-response pairs
"""

import os
import sys
import argparse
import time
import warnings
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model components from pretrain_mlx
from train_pretrain_mlx import (
    MiniMindMLXConfig,
    MiniMindForCausalLM,
    setup_seed,
    get_lr,
    save_checkpoint,
    load_checkpoint
)


class SFTDataset:
    """Dataset for supervised fine-tuning"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
        
        # Token sequences for identifying assistant responses
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
    
    def _load_data(self):
        """Load JSONL data with instruction-response pairs"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def _generate_loss_mask(self, input_ids):
        """Generate loss mask - only compute loss on assistant responses"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # Find <bos>assistant token sequence
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # Find <eos> token sequence
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # Mask tokens between start and end (assistant response)
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats
        if 'conversations' in item:
            # Format: {"conversations": [{"role": "user", "content": "..."}, ...]}
            conversation = item['conversations']
        else:
            # Format: {"instruction": "...", "output": "..."}
            conversation = [{"role": "user", "content": item.get('instruction', '')}]
            if 'output' in item:
                conversation.append({"role": "assistant", "content": item['output']})
        
        # Format with chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        input_ids_list = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids_list += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids_list))
        
        # Generate dynamic loss mask
        loss_mask = self._generate_loss_mask(input_ids_list)
        
        # Build training data (shift by 1 for next-token prediction)
        input_ids = np.array(input_ids_list[:-1], dtype=np.int32)
        labels = np.array(input_ids_list[1:], dtype=np.int32)
        loss_mask = np.array(loss_mask[1:], dtype=np.float32)
        
        return input_ids, labels, loss_mask


def collate_fn(batch):
    """Collate function for dataloader"""
    input_ids, labels, loss_masks = zip(*batch)
    return (
        mx.array(np.stack(input_ids)),
        mx.array(np.stack(labels)),
        mx.array(np.stack(loss_masks))
    )


def loss_fn(model, input_ids, labels, loss_mask):
    """
    Compute cross-entropy loss with masking
    
    Args:
        model: The model
        input_ids: Input token IDs [batch_size, seq_len]
        labels: Target token IDs [batch_size, seq_len]
        loss_mask: Mask for loss computation [batch_size, seq_len]
        
    Returns:
        Loss value
    """
    logits = model(input_ids)
    
    # Flatten for loss computation
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    loss_mask_flat = loss_mask.reshape(-1)
    
    # Compute cross-entropy loss
    losses = nn.losses.cross_entropy(logits_flat, labels_flat, reduction='none')
    
    # Apply mask and compute mean
    masked_loss = losses * loss_mask_flat
    loss = masked_loss.sum() / (loss_mask_flat.sum() + 1e-8)  # Avoid division by zero
    
    return loss


def train_epoch(
    model,
    optimizer,
    dataset,
    epoch: int,
    args,
    skip_steps: int = 0
):
    """Train for one epoch"""
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}/{args.epochs}")
    print(f"{'='*80}")
    
    num_batches = len(dataset) // args.batch_size
    total_loss = 0.0
    start_time = time.time()
    
    for step in range(num_batches):
        # Skip steps if resuming
        if step < skip_steps:
            continue
        
        # Get batch
        batch_start = step * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(dataset))
        batch_data = [dataset[i] for i in range(batch_start, batch_end)]
        input_ids, labels, loss_mask = collate_fn(batch_data)
        
        # Update learning rate
        current_step = epoch * num_batches + step
        total_steps = args.epochs * num_batches
        lr = get_lr(current_step, total_steps, args.learning_rate)
        optimizer.learning_rate = lr
        
        # Forward and backward
        loss, grads = mx.value_and_grad(loss_fn)(model, input_ids, labels, loss_mask)
        
        # Gradient clipping - handle nested structure
        def compute_grad_norm(grads):
            """Compute gradient norm recursively"""
            norm_sq = 0.0
            if isinstance(grads, dict):
                for v in grads.values():
                    norm_sq += compute_grad_norm(v)
            elif isinstance(grads, list):
                for item in grads:
                    norm_sq += compute_grad_norm(item)
            else:
                norm_sq += mx.sum(grads * grads).item()
            return norm_sq
        
        def clip_grads(grads, scale):
            """Clip gradients recursively"""
            if isinstance(grads, dict):
                return {k: clip_grads(v, scale) for k, v in grads.items()}
            elif isinstance(grads, list):
                return [clip_grads(item, scale) for item in grads]
            else:
                return grads * scale
        
        grad_norm = compute_grad_norm(grads) ** 0.5
        if grad_norm > args.grad_clip:
            scale = args.grad_clip / grad_norm
            grads = clip_grads(grads, scale)
        
        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Logging
        if (step + 1) % args.log_interval == 0 or step == num_batches - 1:
            avg_loss = total_loss / (step + 1)
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * args.batch_size * args.max_seq_len / elapsed
            eta_min = (elapsed / (step + 1) * num_batches - elapsed) / 60
            
            print(f"Step [{step+1}/{num_batches}] "
                  f"Loss: {avg_loss:.6f} "
                  f"LR: {lr:.10f} "
                  f"Tokens/s: {tokens_per_sec:.0f} "
                  f"ETA: {eta_min:.1f}min")
        
        # Save checkpoint (model weights and resume state)
        if (step + 1) % args.save_interval == 0 or step == num_batches - 1:
            save_checkpoint(model, optimizer, epoch, step, args, save_resume=False)
            save_checkpoint(model, optimizer, epoch, step, args, save_resume=True)
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="MiniMind Full SFT with MLX")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft_mlx', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size (smaller for SFT)")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="初始学习率 (lower for SFT)")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数 (MLX不需要)")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="SFT数据路径")
    parser.add_argument('--from_weight', default='pretrain_mlx', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="Tokenizer路径")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MiniMind Full SFT with MLX")
    print("Optimized for Apple Silicon")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  Layers: {args.num_hidden_layers}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Max Seq Len: {args.max_seq_len}")
    print(f"  Epochs: {args.epochs}")
    print(f"  From Weight: {args.from_weight}")
    print("=" * 80)
    
    # Setup
    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check for resume checkpoint
    resume_state = None
    if args.from_resume == 1:
        moe_suffix = '_moe' if args.use_moe else ''
        resume_path = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}{moe_suffix}_resume.npz'
        if os.path.exists(resume_path):
            print(f"✓ Found resume checkpoint: {resume_path}")
        else:
            print(f"⚠ No resume checkpoint found at {resume_path}, starting from scratch")
    
    # Create config and model
    config = MiniMindMLXConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    
    model = MiniMindForCausalLM(config)
    
    # Load checkpoint based on priority: resume > from_weight
    if args.from_resume == 1:
        moe_suffix = '_moe' if args.use_moe else ''
        resume_path = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}{moe_suffix}_resume.npz'
        if os.path.exists(resume_path):
            resume_state = load_checkpoint(model, resume_path, load_training_state=True)
    elif args.from_weight != 'none':
        moe_suffix = '_moe' if args.use_moe else ''
        ckp_path = f'{args.save_dir}/{args.from_weight}_{args.hidden_size}{moe_suffix}.npz'
        load_checkpoint(model, ckp_path)
    
    # Count parameters
    def count_params(params):
        """Recursively count parameters"""
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, list):
            for item in params:
                total += count_params(item)
        elif hasattr(params, 'size'):
            total += params.size
        return total
    
    num_params = count_params(model.parameters()) / 1e6
    print(f"\n✓ Model initialized: {num_params:.2f}M parameters")
    
    # Load tokenizer and dataset
    print(f"✓ Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    print(f"✓ Loading SFT dataset from {args.data_path}...")
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Create optimizer
    initial_lr = args.learning_rate
    if resume_state and 'learning_rate' in resume_state:
        initial_lr = resume_state['learning_rate']
    optimizer = optim.AdamW(learning_rate=initial_lr)
    
    # Determine starting epoch and step
    start_epoch = resume_state['epoch'] if resume_state else 0
    start_step = resume_state['step'] if resume_state else 0
    
    # Training loop
    print(f"\n{'='*80}")
    if resume_state:
        print(f"Resuming SFT Training from Epoch {start_epoch + 1}, Step {start_step + 1}")
    else:
        print("Starting SFT Training")
    print(f"{'='*80}")
    
    for epoch in range(start_epoch, args.epochs):
        # Skip steps if resuming mid-epoch
        if epoch == start_epoch and start_step > 0:
            print(f"\n⚠ Skipping first {start_step} steps of epoch {epoch + 1}")
            avg_loss = train_epoch(model, optimizer, dataset, epoch, args, skip_steps=start_step)
        else:
            avg_loss = train_epoch(model, optimizer, dataset, epoch, args)
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_loss:.6f}")
        
        # Save resume checkpoint at end of each epoch
        save_checkpoint(model, optimizer, epoch, 0, args, save_resume=True)
    
    print(f"\n{'='*80}")
    print("SFT Training Complete!")
    print(f"{'='*80}")
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
