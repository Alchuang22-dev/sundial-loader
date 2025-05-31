import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from safetensors import safe_open
from ..config.configuration import BaseSeqConfig

class SequenceModel(ABC, nn.Module):
    def __init__(self, config: BaseSeqConfig):
        super().__init__()
        self.config = config
        self.build_layers()

    def preprocess(self, series: Union[torch.Tensor, Any]) -> torch.Tensor:
        """预处理时序数据"""
        if not isinstance(series, torch.Tensor):
            series = torch.tensor(series, dtype=torch.float32)
        
        # 确保维度正确 [batch, seq_len] 或 [seq_len]
        if series.dim() == 1:
            series = series.unsqueeze(0)  # [1, seq_len]
        
        # 截断或填充到指定长度
        target_len = self.config.input_token_len
        if series.size(-1) > target_len:
            series = series[..., :target_len]
        elif series.size(-1) < target_len:
            pad_size = target_len - series.size(-1)
            series = torch.nn.functional.pad(series, (0, pad_size))
            
        return series

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播，返回logits"""
        pass

    def postprocess(self, logits: torch.Tensor) -> torch.Tensor:
        """后处理，默认返回第一个输出长度的预测"""
        target_len = self.config.output_token_lens[0]
        return logits[..., :target_len]

    @abstractmethod
    def build_layers(self):
        """构建模型层"""
        pass

    # 模型注册机制
    _registry: Dict[str, "type[SequenceModel]"] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(subclass):
            cls._registry[name] = subclass
            return subclass
        return wrapper

    @classmethod
    def from_config(cls, config: BaseSeqConfig) -> "SequenceModel":
        model_cls = cls._registry.get(config.model_type)
        if model_cls is None:
            raise ValueError(f"Unknown model type: {config.model_type}")
        return model_cls(config)

    def load_weights(self, weights_path: str):
        """加载safetensors权重"""
        state_dict = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        # 加载权重，允许部分匹配
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")