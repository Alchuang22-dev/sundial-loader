# 确保模型类被导入和注册
from .base_model import SequenceModel
from .timerxl import TimerXL  
from .sundial import Sundial  

__all__ = ["SequenceModel", "TimerXL", "Sundial"]