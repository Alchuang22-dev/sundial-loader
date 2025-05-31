import torch
import torch.nn as nn
import math
from typing import Optional

class RoPEEmbedding(nn.Module):
    """旋转位置编码 - 修复版本"""
    def __init__(self, dim: int, max_seq_len: int = 10000, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
        # 确保维度是偶数
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
            
        # 生成频率，只需要 dim//2 个频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int):
        # x shape: [batch, seq_len, num_heads, head_dim] 或 [batch, num_heads, seq_len, head_dim]
        device = x.device
        dtype = x.dtype
        
        # 生成位置索引
        seq_idx = torch.arange(seq_len, device=device, dtype=dtype)
        
        # 计算频率矩阵 [seq_len, dim//2]
        freqs = torch.outer(seq_idx, self.inv_freq)
        
        # 生成cos和sin [seq_len, dim//2]
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 扩展到完整维度 [seq_len, dim]
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        # 应用旋转变换
        return self._apply_rotary_pos_emb(x, cos, sin)
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x shape: [batch, seq_len, num_heads, head_dim] 或 [batch, num_heads, seq_len, head_dim]
        
        # 获取x的形状信息
        if x.dim() == 4:
            if x.shape[1] == cos.shape[0]:  # [batch, seq_len, num_heads, head_dim]
                seq_dim = 1
                head_dim = x.shape[-1]
            else:  # [batch, num_heads, seq_len, head_dim]
                seq_dim = 2
                head_dim = x.shape[-1]
        else:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
        
        # 确保cos和sin的维度匹配
        target_shape = list(x.shape)
        cos_sin_shape = [1] * len(target_shape)
        cos_sin_shape[seq_dim] = cos.shape[0]  # seq_len
        cos_sin_shape[-1] = head_dim  # head_dim
        
        cos = cos[:target_shape[seq_dim], :head_dim].view(cos_sin_shape)
        sin = sin[:target_shape[seq_dim], :head_dim].view(cos_sin_shape)
        
        # 分离x为两部分进行旋转
        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]
        
        # 应用旋转变换
        cos1 = cos[..., :head_dim//2]
        cos2 = cos[..., head_dim//2:]
        sin1 = sin[..., :head_dim//2]
        sin2 = sin[..., head_dim//2:]
        
        # RoPE变换：[x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x1 = x1 * cos1 - x2 * sin1
        rotated_x2 = x1 * sin2 + x2 * cos2
        
        return torch.cat([rotated_x1, rotated_x2], dim=-1)

class BaseTimeAttention(nn.Module):
    """基础时间感知注意力 - 修复版本"""
    def __init__(self, hidden_size: int, num_heads: int, rope_theta: float = 10000.0, dropout_rate: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout_rate = dropout_rate
        
        assert self.hidden_size % self.num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # 使用正确的head_dim
        self.rope = RoPEEmbedding(self.head_dim, theta=rope_theta)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        print(f"Attention初始化: hidden_size={hidden_size}, num_heads={num_heads}, head_dim={self.head_dim}")
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # 投影到 Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        print(f"QKV shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # 应用RoPE - 在[batch, seq_len, num_heads, head_dim]格式下
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        # 重排维度到 [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores += attention_mask
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)

class TimerAttention(BaseTimeAttention):
    """Timer模型专用的注意力"""
    def __init__(self, config):
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            dropout_rate=getattr(config, 'attention_dropout', 0.0)
        )

class SundialAttention(BaseTimeAttention):
    """Sundial模型专用的注意力"""
    def __init__(self, config):
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            dropout_rate=getattr(config, 'dropout_rate', 0.0)
        )