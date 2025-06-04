import torch
import torch.nn as nn
from typing import Optional

from ..base_model import SequenceModel
from .attention import TimerAttention
from .configuration import TimerConfig

class TimerBlock(nn.Module):
    """Timer Transformer Block"""
    def __init__(self, config: TimerConfig):
        super().__init__()
        self.attention = TimerAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 注意力 + 残差连接
        attn_out = self.attention(self.ln1(x), attention_mask)
        x = x + attn_out
        
        # 前馈网络 + 残差连接
        ff_out = self.feed_forward(self.ln2(x))
        x = x + ff_out
        
        return x

@SequenceModel.register("timer")
class TimerXL(SequenceModel):
    config_class = TimerConfig  # 指定配置类
    
    def __init__(self, config: TimerConfig):
        super().__init__(config)
        
    def build_layers(self):
        # 输入嵌入层
        self.input_embedding = nn.Linear(1, self.config.hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TimerBlock(self.config) for _ in range(self.config.num_hidden_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(self.config.hidden_size)
        self.output_head = nn.Linear(
            self.config.hidden_size, 
            self.config.output_token_lens[0]
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.input_embedding(input_ids.unsqueeze(-1))
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 输出
        x = self.output_norm(x)
        last_hidden = x[:, -1, :]
        predictions = self.output_head(last_hidden)
        
        return predictions