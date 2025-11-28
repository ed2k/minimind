"""
MiniMind Model Inference using MLX
Optimized for Apple Silicon (M1/M2/M3/M4)
"""

import os
import argparse
import random
import warnings
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


def setup_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


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
        inference_rope_scaling: bool = False,
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
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
        self.inference_rope_scaling = inference_rope_scaling
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        
        # Derived values
        self.head_dim = hidden_size // num_attention_heads


def load_weights_to_mlx(weight_path: str, config: MiniMindMLXConfig) -> Dict:
    """
    Load weights and convert to MLX format
    Supports both .npz (MLX native) and .pth (PyTorch) formats
    
    Args:
        weight_path: Path to weight file (.npz or .pth)
        config: Model configuration
        
    Returns:
        Dictionary of MLX arrays
    """
    import os
    
    if weight_path.endswith('.npz'):
        # Load MLX native format
        print(f"Loading MLX weights from {weight_path}...")
        flat_weights = mx.load(weight_path)
        
        # Reconstruct nested structure from flattened weights
        def unflatten_params(flat_dict):
            """Reconstruct nested structure from flat dict"""
            nested = {}
            for key, value in flat_dict.items():
                parts = key.split('.')
                current = nested
                
                for i, part in enumerate(parts[:-1]):
                    if part.isdigit():
                        idx = int(part)
                        if not isinstance(current, list):
                            current = []
                        while len(current) <= idx:
                            current.append({})
                        if not isinstance(current[idx], dict):
                            current[idx] = {}
                        current = current[idx]
                    else:
                        if part not in current:
                            if i + 1 < len(parts) - 1 and parts[i + 1].isdigit():
                                current[part] = []
                            else:
                                current[part] = {}
                        current = current[part]
                
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
        
        return unflatten_params(flat_weights)
        
    elif weight_path.endswith('.pth'):
        # Load PyTorch format and convert
        print(f"Loading PyTorch weights from {weight_path}...")
        import torch
        state_dict = torch.load(weight_path, map_location='cpu')
        
        # Convert to MLX format (flat structure)
        mlx_weights = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                mlx_weights[key] = mx.array(value.cpu().numpy())
            else:
                mlx_weights[key] = value
        
        return mlx_weights
    else:
        raise ValueError(f"Unsupported weight format: {weight_path}. Use .npz or .pth")


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
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int):
        """Generate cos and sin for rotary embeddings"""
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors"""
    def rotate_half(x):
        x1, x2 = mx.split(x, 2, axis=-1)
        return mx.concatenate([-x2, x1], axis=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MiniMindAttention(nn.Module):
    """Multi-head attention with GQA support"""
    def __init__(self, config: MiniMindMLXConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_key_value_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        
        # Project to Q, K, V
        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(L)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        
        # Repeat K, V for GQA
        if self.n_rep > 1:
            keys = mx.repeat(keys, self.n_rep, axis=1)
            values = mx.repeat(values, self.n_rep, axis=1)
        
        # Scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.head_dim))
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = mx.softmax(scores, axis=-1)
        output = attn_weights @ values
        
        # Reshape and project
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMindMLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config: MiniMindMLXConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

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

    def __call__(self, x, mask=None, cache=None):
        # Self-attention with residual
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        
        # MLP with residual
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        
        return out


class MiniMindForCausalLM(nn.Module):
    """MiniMind model for causal language modeling in MLX"""
    def __init__(self, config: MiniMindMLXConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = [MiniMindDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        
        # Final norm and output
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids, mask=None):
        # Embed tokens
        x = self.embed_tokens(input_ids)
        
        # Create causal mask if not provided
        if mask is None and input_ids.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
            mask = mask.astype(x.dtype)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final norm and project to vocabulary
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.85,
        top_p: float = 0.85,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> mx.array:
        """
        Generate text using the model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            
        Returns:
            Generated token IDs
        """
        eos_token_id = eos_token_id or self.config.eos_token_id
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self(input_ids)
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p (nucleus) sampling
            sorted_logits = mx.sort(next_token_logits, axis=-1)[:, ::-1]
            sorted_indices = mx.argsort(next_token_logits, axis=-1)[:, ::-1]
            
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1]
            sorted_indices_to_remove[:, 0] = False
            
            # Set logits to -inf for removed tokens
            next_token_logits = mx.where(
                sorted_indices_to_remove,
                mx.array(-float('inf')),
                next_token_logits
            )
            
            # Sample from the distribution
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs), axis=-1)
            
            # Append to input_ids
            input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)
            
            # Check for EOS
            if eos_token_id is not None and next_token[0].item() == eos_token_id:
                break
        
        return input_ids


def init_model(args):
    """Initialize model and tokenizer"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    # Create config
    config = MiniMindMLXConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling
    )
    
    # Create model
    model = MiniMindForCausalLM(config)
    
    # Load weights if using native weights
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        
        # Try .npz first (MLX native), then .pth (PyTorch)
        npz_path = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.npz'
        pth_path = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        if os.path.exists(npz_path):
            ckp = npz_path
        elif os.path.exists(pth_path):
            ckp = pth_path
        else:
            raise FileNotFoundError(f"No weights found at {npz_path} or {pth_path}")
        
        weights = load_weights_to_mlx(ckp, config)
        model.update(weights)
        print("✓ Weights loaded successfully!")
    
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
    print(f'MiniMind模型参数: {num_params:.2f} M(illion)')
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话 (MLX版本)")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft_mlx', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（暂不支持）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MiniMind MLX Inference (Optimized for Apple Silicon)")
    print("=" * 80)
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0:
            print(f'👶: {prompt}')
        
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        
        # Prepare input
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason':
            templates["enable_thinking"] = True
        
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        input_ids = tokenizer(inputs, return_tensors="np")["input_ids"]
        input_ids = mx.array(input_ids)
        
        print('🤖️: ', end='', flush=True)
        
        # Generate
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode response
        response_ids = generated_ids[0, input_ids.shape[1]:].tolist()
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        print(response)
        
        conversation.append({"role": "assistant", "content": response})
        print('\n')


if __name__ == "__main__":
    main()
