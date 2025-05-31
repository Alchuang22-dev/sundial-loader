import torch
import torch.nn as nn
import math
from typing import Optional

from .base_model import SequenceModel
from .attention import SundialAttention  # 使用专门的SundialAttention
from ..config.configuration import SundialConfig

class DiffusionTimeStep(nn.Module):
    """扩散时间步嵌入"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, timesteps: torch.Tensor):
        # 时间步编码（类似Transformer的位置编码）
        half_dim = self.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class FlowLayer(nn.Module):
    """Flow变换层"""
    def __init__(self, config: SundialConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.flow_depth = config.flow_loss_depth
        
        # Flow变换网络
        self.flow_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, x: torch.Tensor, reverse: bool = False):
        if reverse:
            # 逆Flow变换
            return x - self.flow_net(x)
        else:
            # 正Flow变换
            return x + self.flow_net(x)

class SundialBlock(nn.Module):
    """Sundial Transformer Block with Diffusion-Flow"""
    def __init__(self, config: SundialConfig):
        super().__init__()
        self.config = config
        
        # 使用Sundial专用的注意力
        self.attention = SundialAttention(config)
        
        # Flow层
        self.flow_layer = FlowLayer(config)
        
        # 标准FFN
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout_rate)
        )
        
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ln3 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: torch.Tensor, timestep_emb: Optional[torch.Tensor] = None):
        # 时间注意力
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        
        # Flow变换
        if timestep_emb is not None:
            x = x + timestep_emb.unsqueeze(1)  # 广播时间步嵌入
        flow_out = self.flow_layer(self.ln2(x))
        x = x + flow_out
        
        # FFN
        ff_out = self.feed_forward(self.ln3(x))
        x = x + ff_out
        
        return x

@SequenceModel.register("sundial")
class Sundial(SequenceModel):
    def __init__(self, config: SundialConfig):
        super().__init__(config)
        
    def build_layers(self):
        # 输入嵌入
        self.input_embedding = nn.Linear(1, self.config.hidden_size)
        
        # 扩散时间步嵌入
        self.timestep_embedding = DiffusionTimeStep(self.config.hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([
            SundialBlock(self.config) for _ in range(self.config.num_hidden_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(self.config.hidden_size)
        self.output_head = nn.Linear(
            self.config.hidden_size,
            self.config.output_token_lens[0]
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # 输入嵌入
        x = self.input_embedding(input_ids.unsqueeze(-1))
        
        # 扩散采样循环
        for step in range(self.config.num_sampling_steps):
            # 生成时间步
            timesteps = torch.full((batch_size,), step, device=input_ids.device)
            timestep_emb = self.timestep_embedding(timesteps)
            
            # 通过Transformer层
            for layer in self.layers:
                x = layer(x, timestep_emb)
        
        # 输出
        x = self.output_norm(x)
        last_hidden = x[:, -1, :]
        predictions = self.output_head(last_hidden)
        
        return predictions