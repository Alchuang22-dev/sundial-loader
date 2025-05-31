import torch
import torch.nn as nn
from typing import Optional

from .base_model import SequenceModel
from .attention import TimerAttention
from ..config.configuration import TimerConfig

class TimerBlock(nn.Module):
    """Timer Transformer Block"""
    def __init__(self, config: TimerConfig):
        super().__init__()
        self.attention = TimerAttention(config)  # 使用专门的TimerAttention
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
    def __init__(self, config: TimerConfig):
        super().__init__(config)
        
    def build_layers(self):
        # 输入嵌入层：将时序值映射到隐藏维度
        self.input_embedding = nn.Linear(1, self.config.hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TimerBlock(self.config) for _ in range(self.config.num_hidden_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(self.config.hidden_size)
        self.output_head = nn.Linear(
            self.config.hidden_size, 
            self.config.output_token_lens[0]  # 预测长度
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # 嵌入：[batch_size, seq_len, 1] -> [batch_size, seq_len, hidden_size]
        x = self.input_embedding(input_ids.unsqueeze(-1))
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 输出归一化
        x = self.output_norm(x)
        
        # 预测：使用最后一个时间步的表示
        last_hidden = x[:, -1, :]  # [batch_size, hidden_size]
        predictions = self.output_head(last_hidden)  # [batch_size, pred_len]
        
        return predictions