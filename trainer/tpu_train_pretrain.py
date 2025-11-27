import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
from contextlib import nullcontext
from torch import optim, nn
from torch.utils.data import DataLoader

# TPU/XLA imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
# Note: XLA autocast can cause dtype mismatches, use explicit casting instead
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, lm_checkpoint, setup_seed, init_model, SkipBatchSampler

# TPU-specific helper functions
def is_main_process():
    """Check if this is the main process (ordinal 0)"""
    return xr.global_ordinal() == 0

def init_tpu_distributed():
    """Initialize TPU distributed training"""
    return xr.global_ordinal()

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, use_mixed_precision=False, target_dtype=torch.float32):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)
        
        # Note: Don't cast X to bfloat16 - it contains token IDs (integers)
        # The embedding layer will handle the conversion internally
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            # Gradient clipping for XLA
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # XLA optimizer step - this marks the step for execution
            xm.optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            # Mark step for XLA to execute accumulated operations
            xm.mark_step()
            
            spend_time = time.time() - start_time
            # Use xm.mesh_reduce to get loss across all TPU cores
            current_loss = xm.mesh_reduce('loss', loss.item() * args.accumulation_steps, lambda x: sum(x) / len(x))
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            if is_main_process():
                Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
                if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # Ensure all operations are complete before saving
            xm.mark_step()
            xm.wait_device_ops()
            
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # XLA models don't use DDP wrapper, get state_dict directly
            state_dict = model.state_dict()
            # Convert to CPU and half precision for saving
            state_dict = {k: v.cpu().half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=None, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="xla", help="训练设备 (TPU使用xla)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化TPU环境和随机种子 ==========
    local_rank = init_tpu_distributed()
    device = torch_xla.device()  # Get TPU device
    setup_seed(42 + xr.global_ordinal())
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 (TPU XLA) ==========
    # For TPU, we'll cast the model explicitly instead of using autocast
    # to avoid dtype mismatches in attention operations
    use_mixed_precision = args.dtype in ['bfloat16', 'float16']
    target_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext()  # Don't use autocast on TPU
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=device)
    
    # Cast model to target dtype for mixed precision on TPU
    if use_mixed_precision:
        model = model.to(target_dtype)
        if is_main_process():
            print(f"Model cast to {target_dtype} for mixed precision training")
    
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # XLA uses distributed sampler automatically with ParallelLoader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=True
    )
    scaler = None  # XLA handles mixed precision internally
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        # scaler not used in XLA
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. XLA不需要DDP包装 ==========
    # XLA handles distributed training automatically, no DDP wrapper needed
    # Model is already on the correct XLA device
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler, args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers)
            # Wrap with XLA ParallelLoader for efficient TPU data loading
            para_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
            if is_main_process():
                Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, para_loader, len(loader) + start_step + 1, start_step, wandb, use_mixed_precision, target_dtype)
        else: # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
            # Wrap with XLA ParallelLoader for efficient TPU data loading
            para_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
            train_epoch(epoch, para_loader, len(loader), 0, wandb, use_mixed_precision, target_dtype)
    
    # ========== 9. 训练完成，同步所有TPU核心 ==========
    xm.rendezvous('training_complete')
