# 确保模型类被导入和注册
from .base_model import SequenceModel

from .base_model import SequenceModel

# 导入各个模型以触发注册
from .timerxl import TimerXL, TimerConfig
from .sundial import Sundial, SundialConfig

__all__ = [
    "SequenceModel", 
    "TimerXL", "TimerConfig",
    "Sundial", "SundialConfig"
]