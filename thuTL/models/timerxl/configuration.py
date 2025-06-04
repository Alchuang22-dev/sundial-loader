from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
from pathlib import Path

@dataclass
class TimerConfig:
    """TimerXL 模型配置"""
    # 模型标识
    model_type: str = "timer"
    
    # 核心架构参数
    hidden_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    
    # 时序相关参数
    input_token_len: int = 96
    output_token_lens: List[int] = field(default_factory=lambda: [96])
    
    # 注意力参数
    attention_dropout: float = 0.0
    rope_theta: int = 10000
    
    # 前馈网络参数
    intermediate_size: int = 2048
    hidden_act: str = "silu"
    
    # 训练参数
    initializer_range: float = 0.02
    max_position_embeddings: int = 10000
    torch_dtype: str = "float32"
    use_cache: bool = True
    
    # 扩展字段，支持未来参数
    _extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, path: str) -> "TimerConfig":
        """从JSON文件加载配置"""
        config_data = json.loads(Path(path).read_text())
        
        # 获取已声明的字段
        declared_fields = set(cls.__dataclass_fields__.keys())
        
        # 分离已知和未知字段
        known = {}
        extras = {}
        for k, v in config_data.items():
            if k in declared_fields and not k.startswith('_'):
                known[k] = v
            else:
                extras[k] = v
        
        known['_extras'] = extras
        return cls(**known)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            if not field_name.startswith('_'):
                result[field_name] = getattr(self, field_name)
        
        # 添加扩展字段
        result.update(self._extras)
        return result
    
    def save_json(self, path: str):
        """保存为JSON文件"""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))