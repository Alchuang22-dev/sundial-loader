from dataclasses import dataclass, field
from typing import List, Optional, Type, Dict, Any
import json
from pathlib import Path

@dataclass
class BaseSeqConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    input_token_len: int
    output_token_lens: List[int]
    
    # 通用字段
    hidden_act: str = "silu"
    intermediate_size: int = 2048
    max_position_embeddings: int = 10000
    rope_theta: int = 10000
    torch_dtype: str = "float32"
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # 未知字段
    _extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, path: str) -> "BaseSeqConfig":
        config_data = json.loads(Path(path).read_text())
        model_type = config_data.get("model_type")
        
        # 修正：应该先确定子类再处理字段
        mapping = {
            "sundial": SundialConfig,
            "timer": TimerConfig,
        }
        config_cls = mapping.get(model_type, cls)
        
        # 获取该类所有字段（包括继承的）
        declared_fields = set()
        for cls_in_mro in config_cls.__mro__:
            if hasattr(cls_in_mro, '__dataclass_fields__'):
                declared_fields.update(cls_in_mro.__dataclass_fields__.keys())
        
        # 分离已知和未知字段
        known = {}
        extras = {}
        for k, v in config_data.items():
            if k in declared_fields and not k.startswith('_'):
                known[k] = v
            else:
                extras[k] = v
        
        known['_extras'] = extras
        return config_cls(**known)

@dataclass
class SundialConfig(BaseSeqConfig):
    diffusion_batch_mul: int = 4
    flow_loss_depth: int = 3
    num_sampling_steps: int = 50
    dropout_rate: float = 0.1

@dataclass
class TimerConfig(BaseSeqConfig):
    attention_dropout: float = 0.0