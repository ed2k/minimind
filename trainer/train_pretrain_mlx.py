"""
MiniMind Pretraining using MLX
Optimized for Apple Silicon (M1/M2/M3/M4)

Note: MLX training is still experimental. This implementation provides:
- Single-device training on Apple Silicon
- Automatic differentiation with mlx.nn
- Efficient memory usage with unified memory
- Native bfloat16 support
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


def setup_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def get_lr(step: int, total_steps: int, learning_rate: float, warmup_steps: int = 100) -> float:
    """
    Cosine learning rate schedule with warmup
    
    Args:
        step: Current training step
        total_steps: Total number of training steps
        learning_rate: Maximum learning rate
        warmup_steps: Number of warmup steps
        
    Returns:
        Current learning rate
    """
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return learning_rate * 0.5 * (1 + np.cos(np.pi * progress))


class MiniMindMLXConfig:
    """Configuration for MiniMind model in MLX"""
    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = 'silu',
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1000000.0,
        use_moe: bool = False,
        **kwargs
    ):
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or (hidden_size * 4)
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_moe = use_moe
        self.head_dim = hidden_size // num_attention_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RotaryEmbedding:
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dims: int, max_position_embeddings: int = 32768, base: float = 1000000.0):
        self.dims = dims
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int):
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings"""
    def rotate_half(x):
        x1, x2 = mx.split(x, 2, axis=-1)
        return mx.concatenate([-x2, x1], axis=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MiniMindAttention(nn.Module):
    """Multi-head attention with GQA"""
    def __init__(self, config: MiniMindMLXConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def __call__(self, x, mask=None):
        B, L, _ = x.shape
        
        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        cos, sin = self.rotary_emb(L)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        
        if self.n_rep > 1:
            keys = mx.repeat(keys, self.n_rep, axis=1)
            values = mx.repeat(values, self.n_rep, axis=1)
        
        scores = (queries @ keys.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.head_dim))
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = mx.softmax(scores, axis=-1)
        output = attn_weights @ values
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMindMLP(nn.Module):
    """Feed-forward network with SwiGLU"""
    def __init__(self, config: MiniMindMLXConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMindDecoderLayer(nn.Module):
    """Transformer decoder layer"""
    def __init__(self, config: MiniMindMLXConfig):
        super().__init__()
        self.self_attn = MiniMindAttention(config)
        self.mlp = MiniMindMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, mask=None):
        r = self.self_attn(self.input_layernorm(x), mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class MiniMindForCausalLM(nn.Module):
    """MiniMind model for causal language modeling"""
    def __init__(self, config: MiniMindMLXConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MiniMindDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids, mask=None):
        x = self.embed_tokens(input_ids)
        
        if mask is None and input_ids.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
            mask = mask.astype(x.dtype)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class PretrainDataset:
    """Dataset for pretraining"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
    
    def _load_data(self):
        """Load JSONL data"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item['text'])
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        input_ids = tokens['input_ids'][0]
        attention_mask = tokens['attention_mask'][0]
        
        # For causal LM, labels are input_ids shifted
        labels = input_ids.copy()
        
        return input_ids, labels, attention_mask


def collate_fn(batch):
    """Collate function for dataloader"""
    input_ids, labels, attention_masks = zip(*batch)
    return (
        mx.array(np.stack(input_ids)),
        mx.array(np.stack(labels)),
        mx.array(np.stack(attention_masks))
    )


def loss_fn(model, input_ids, labels, loss_mask):
    """
    Compute cross-entropy loss
    
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
    loss = masked_loss.sum() / loss_mask_flat.sum()
    
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


def save_checkpoint(model, optimizer, epoch, step, args, save_resume=False):
    """Save model checkpoint"""
    os.makedirs(args.save_dir, exist_ok=True)
    
    moe_suffix = '_moe' if args.use_moe else ''
    
    if save_resume:
        # Save resume checkpoint with training state
        ckp_path = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}{moe_suffix}_resume.npz'
    else:
        # Save model weights only
        ckp_path = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}{moe_suffix}.npz'
    
    # Flatten nested parameters structure for saving
    def flatten_params(params, prefix=''):
        """Flatten nested dict/list structure into flat dict"""
        flat = {}
        if isinstance(params, dict):
            for k, v in params.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                flat.update(flatten_params(v, new_prefix))
        elif isinstance(params, list):
            for i, item in enumerate(params):
                new_prefix = f"{prefix}.{i}" if prefix else str(i)
                flat.update(flatten_params(item, new_prefix))
        else:
            # It's an array
            flat[prefix] = params
        return flat
    
    # Get flattened weights
    weights = flatten_params(model.parameters())
    
    if save_resume:
        # Save with training state for resuming
        # Note: MLX doesn't have direct optimizer state serialization,
        # so we save epoch and step info separately
        weights['__epoch__'] = mx.array([epoch])
        weights['__step__'] = mx.array([step])
        weights['__learning_rate__'] = mx.array([optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, 'item') else optimizer.learning_rate])
    
    # Save using mx.savez
    mx.savez(ckp_path, **weights)
    if save_resume:
        print(f"✓ Resume checkpoint saved to {ckp_path}")
    else:
        print(f"✓ Checkpoint saved to {ckp_path}")


def load_checkpoint(model, checkpoint_path, load_training_state=False):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None if load_training_state else None
    
    # Load flattened weights
    flat_weights = mx.load(checkpoint_path)
    
    # Reconstruct nested structure
    def unflatten_params(flat_dict):
        """Reconstruct nested structure from flat dict"""
        nested = {}
        for key, value in flat_dict.items():
            parts = key.split('.')
            current = nested
            
            for i, part in enumerate(parts[:-1]):
                # Check if this should be a list index
                if part.isdigit():
                    idx = int(part)
                    if not isinstance(current, list):
                        # Convert to list if needed
                        current = []
                    # Extend list if needed
                    while len(current) <= idx:
                        current.append({})
                    if not isinstance(current[idx], dict):
                        current[idx] = {}
                    current = current[idx]
                else:
                    if part not in current:
                        # Look ahead to see if next part is a digit
                        if i + 1 < len(parts) - 1 and parts[i + 1].isdigit():
                            current[part] = []
                        else:
                            current[part] = {}
                    current = current[part]
            
            # Set the final value
            final_key = parts[-1]
            if final_key.isdigit():
                idx = int(final_key)
                if not isinstance(current, list):
                    current = []
                while len(current) <= idx:
                    current.append(None)
                current[idx] = value
            else:
                current[final_key] = value
        
        return nested
    
    # Extract training state if present
    training_state = None
    if load_training_state:
        training_state = {}
        if '__epoch__' in flat_weights:
            training_state['epoch'] = int(flat_weights['__epoch__'].item())
            del flat_weights['__epoch__']
        if '__step__' in flat_weights:
            training_state['step'] = int(flat_weights['__step__'].item())
            del flat_weights['__step__']
        if '__learning_rate__' in flat_weights:
            training_state['learning_rate'] = float(flat_weights['__learning_rate__'].item())
            del flat_weights['__learning_rate__']
    
    nested_weights = unflatten_params(flat_weights)
    model.update(nested_weights)
    
    if load_training_state and training_state:
        print(f"✓ Resume checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {training_state.get('epoch', 0)}, step {training_state.get('step', 0)}")
        return training_state
    else:
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="MiniMind Pretraining with MLX")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_mlx', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size (smaller for MLX)")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数 (MLX不需要)")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="Tokenizer路径")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MiniMind Pretraining with MLX")
    print("Optimized for Apple Silicon")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  Layers: {args.num_hidden_layers}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Max Seq Len: {args.max_seq_len}")
    print(f"  Epochs: {args.epochs}")
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
        """Recursively count parameters in nested dict/list structure"""
        total = 0
        if isinstance(params, dict):
            for k, v in params.items():
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
    
    print(f"✓ Loading dataset from {args.data_path}...")
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
        print(f"Resuming Training from Epoch {start_epoch + 1}, Step {start_step + 1}")
    else:
        print("Starting Training")
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
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
