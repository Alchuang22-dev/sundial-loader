import torch
from pathlib import Path
from typing import Union, Optional

from ..config.configuration import BaseSeqConfig
from ..models.base_model import SequenceModel
from ..utils.model_loader import ModelDownloader

# 确保模型被导入和注册
from ..models.timerxl import TimerXL
from ..models.sundial import Sundial

class LLM:
    def __init__(self, model: Union[str, Path], task: str = "generate", 
                 cache_dir: Optional[str] = None):
        self.task = task
        self.model_name = str(model)
        
        # 如果是本地路径
        if Path(model).exists():
            config_path = Path(model) / "config.json"
            weights_path = Path(model) / "model.safetensors"
        else:
            # 从HuggingFace下载
            downloader = ModelDownloader(cache_dir or "./models")
            
            # 修正：根据模型名称选择下载方法
            if model.lower() in ["timer", "timerxl"]:
                files = downloader.download_timerxl()
            elif model.lower() in ["sundial"]:
                files = downloader.download_sundial()
            else:
                raise ValueError(f"Unknown model: {model}")
            
            config_path = files["config.json"]
            weights_path = files["model.safetensors"]
        
        # 加载配置和模型
        self.config = BaseSeqConfig.from_json(str(config_path))
        
        # 调试：打印注册的模型
        print(f"已注册的模型类型: {list(SequenceModel._registry.keys())}")
        print(f"配置中的模型类型: {self.config.model_type}")
        
        self.model = SequenceModel.from_config(self.config)
        self.model.load_weights(str(weights_path))
        self.model.eval()

    def generate(self, inputs, **kwargs):
        """生成预测结果"""
        with torch.no_grad():
            processed_inputs = self.model.preprocess(inputs)
            logits = self.model.forward(processed_inputs, **kwargs)
            outputs = self.model.postprocess(logits)
        return outputs

    def apply_model(self, func):
        """应用函数到模型"""
        return func(self.model)