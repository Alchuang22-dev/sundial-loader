import json
import torch
from pathlib import Path
from typing import Union, Optional, Dict, Any
from urllib.parse import urlparse
import requests
import tempfile
import os

from ..models.base_config import BaseSeqConfig
from ..models.base_model import SequenceModel
from ..utils.model_loader import ModelDownloader

# 确保模型被导入和注册
from ..models.timerxl import TimerXL, TimerConfig
from ..models.sundial import Sundial, SundialConfig

class LLM:
    """
    thuTL 时序大模型推理引擎
    
    支持:
    - 本地模型路径
    - HuggingFace模型名称
    - HTTP/HTTPS URI
    - 固定Task模式（为IoTDB优化）
    """
    
    def __init__(self, 
                 model: Union[str, Path], 
                 task: str = "generate", 
                 cache_dir: Optional[str] = None,
                 uri: Optional[str] = None,
                 **kwargs):
        """
        初始化LLM引擎
        
        Args:
            model: 模型标识符（本地路径、模型名称或URI）
            task: 任务类型，默认"generate"
            cache_dir: 缓存目录
            uri: 模型URI地址（可选，用于从指定URL下载）
            **kwargs: 额外的模型参数
        """
        self.task = task
        self.model_name = str(model)
        self.uri = uri
        self.kwargs = kwargs
        
        # 根据输入类型确定模型来源
        config_path, weights_path = self._resolve_model_source(model, cache_dir)
        
        # 加载配置和模型
        self.config = self._load_config(config_path)
        self.model = self._load_model(weights_path)
        
        print(f"LLM engine initialized {self.model_name} (task: {self.task})")

    def _resolve_model_source(self, model: Union[str, Path], cache_dir: Optional[str]) -> tuple:
        """解析模型来源并返回配置和权重文件路径"""
        
        # 1. 检查是否为URI
        if self.uri or self._is_uri(str(model)):
            return self._download_from_uri(self.uri or str(model), cache_dir)
        
        # 2. 检查是否为本地路径
        if Path(model).exists():
            return self._load_from_local_path(Path(model))
        
        # 3. 从HuggingFace下载
        return self._download_from_huggingface(str(model), cache_dir)
    
    def _is_uri(self, path: str) -> bool:
        """检查是否为URI"""
        try:
            result = urlparse(path)
            return result.scheme in ['http', 'https', 'ftp', 'file']
        except:
            return False
    
    def _download_from_uri(self, uri: str, cache_dir: Optional[str]) -> tuple:
        """从URI下载模型文件"""
        print(f"Downloading model from URI: {uri}")
        
        cache_dir = Path(cache_dir or "./models")
        cache_dir.mkdir(exist_ok=True)
        
        # 解析URI获取模型名称
        parsed = urlparse(uri)
        model_name = Path(parsed.path).stem or "downloaded_model"
        model_dir = cache_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # 下载文件
        files_to_download = ["config.json", "model.safetensors"]
        downloaded_files = {}
        
        for filename in files_to_download:
            file_uri = uri.rstrip('/') + '/' + filename
            local_path = model_dir / filename
            
            try:
                print(f"dowloading {filename} from {file_uri}")
                self._download_file(file_uri, local_path)
                downloaded_files[filename] = local_path
                print(f"{filename} downloaded")
            except Exception as e:
                print(f"Download {filename} failed: {e}")
                raise
        
        return downloaded_files["config.json"], downloaded_files["model.safetensors"]
    
    def _download_file(self, url: str, local_path: Path):
        """下载单个文件"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def _load_from_local_path(self, model_path: Path) -> tuple:
        """从本地路径加载模型"""
        print(f"Loading model from path: {model_path}")
        
        config_path = model_path / "config.json"
        weights_path = model_path / "model.safetensors"
        
        if not config_path.exists():
            raise FileNotFoundError(f"File not exist: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"File not exist: {weights_path}")
        
        return config_path, weights_path
    
    def _download_from_huggingface(self, model_name: str, cache_dir: Optional[str]) -> tuple:
        """从HuggingFace下载模型"""
        print(f"Downloading {model_name} from HuggingFace: ")
        
        downloader = ModelDownloader(cache_dir or "./models")
        
        # 根据模型名称选择下载方法
        if model_name.lower() in ["timer", "timerxl"]:
            files = downloader.download_timerxl()
        elif model_name.lower() in ["sundial"]:
            files = downloader.download_sundial()
        else:
            raise ValueError(f"Can not find {model_name}.")
        
        return files["config.json"], files["model.safetensors"]
    
    def _load_config(self, config_path: Path) -> BaseSeqConfig:
        """加载模型配置"""
        print(f"Loading file: {config_path}")
        
        # 根据模型类型选择配置类
        config_data = json.loads(Path(config_path).read_text())
        model_type = config_data.get("model_type")
        
        if model_type == "timer":
            from ..models.timerxl.configuration import TimerConfig
            return TimerConfig.from_json(str(config_path))
        elif model_type == "sundial":
            from ..models.sundial.configuration import SundialConfig
            return SundialConfig.from_json(str(config_path))
        else:
            # 使用通用配置作为后备
            return BaseSeqConfig.from_json(str(config_path))
    
    def _load_model(self, weights_path: Path) -> SequenceModel:
        """加载模型"""
        print(f"Building model: {self.config.model_type}")
        print(f"Model registered: {list(SequenceModel._registry.keys())}")
        
        model = SequenceModel.from_config(self.config)
        
        print(f"Loading weights: {weights_path}")
        model.load_weights(str(weights_path))
        model.eval()
        
        return model

    def generate(self, inputs, **kwargs):
        """生成预测结果"""
        merged_kwargs = {**self.kwargs, **kwargs}
        
        with torch.no_grad():
            processed_inputs = self.model.preprocess(inputs)
            logits = self.model.forward(processed_inputs, **merged_kwargs)
            outputs = self.model.postprocess(logits)
        return outputs

    def apply_model(self, func):
        """应用函数到模型"""
        return func(self.model)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        def extract_info(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "model_name": self.model_name,
                "model_type": model.config.model_type,
                "task": self.task,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "input_length": model.config.input_token_len,
                "output_length": model.config.output_token_lens[0],
                "hidden_size": model.config.hidden_size,
                "num_layers": model.config.num_hidden_layers,
                "num_heads": model.config.num_attention_heads
            }
        
        return self.apply_model(extract_info)