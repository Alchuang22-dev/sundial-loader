"""
IoTDB 专用的固定Task LLM接口
为Apache IoTDB优化，提供简化的API
"""

import torch
import pandas as pd
from typing import Union, Optional, Dict, Any
from pathlib import Path

from ..engine.llm_engine import LLM

class IoTDBLLM:
    """
    IoTDB专用的LLM接口
    - 固定task为"generate"
    - 简化的API设计
    - 针对时序预测优化
    """
    
    def __init__(self, 
                 model: Union[str, Path], 
                 cache_dir: Optional[str] = None,
                 uri: Optional[str] = None,
                 **model_params):
        """
        初始化IoTDB LLM
        
        Args:
            model: 模型标识符
            cache_dir: 缓存目录
            uri: 模型URI（可选）
            **model_params: 模型参数
        """
        # 固定task为generate
        self.llm = LLM(
            model=model,
            task="generate",  # 固定为generate
            cache_dir=cache_dir,
            uri=uri,
            **model_params
        )
        
        self.model_info = self.llm.get_model_info()
        print(f"🏛️  IoTDB LLM 就绪: {self.model_info['model_name']}")
    
    def forecast(self, 
                time_series: Union[torch.Tensor, list, pd.Series],
                output_length: Optional[int] = None) -> torch.Tensor:
        """
        时序预测
        
        Args:
            time_series: 输入时序数据
            output_length: 输出长度（可选）
            
        Returns:
            预测结果
        """
        # 数据预处理
        if isinstance(time_series, (list, pd.Series)):
            time_series = torch.tensor(time_series, dtype=torch.float32)
        
        # 生成预测
        predictions = self.llm.generate(time_series)
        
        # 如果指定了输出长度，则截取
        if output_length and len(predictions) > output_length:
            predictions = predictions[:output_length]
        
        return predictions
    
    def batch_forecast(self, 
                      batch_series: list,
                      output_length: Optional[int] = None) -> list:
        """
        批量时序预测
        
        Args:
            batch_series: 批量时序数据
            output_length: 输出长度
            
        Returns:
            批量预测结果
        """
        results = []
        for series in batch_series:
            pred = self.forecast(series, output_length)
            results.append(pred)
        return results
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """获取模型能力信息"""
        return {
            "input_length": self.model_info["input_length"],
            "output_length": self.model_info["output_length"],
            "model_type": self.model_info["model_type"],
            "supported_tasks": ["time_series_forecasting"],
            "batch_inference": True,
            "streaming_inference": False
        }

# 便捷函数
def create_timer_llm(cache_dir: Optional[str] = None, 
                    uri: Optional[str] = None) -> IoTDBLLM:
    """创建TimerXL模型的IoTDB LLM"""
    return IoTDBLLM("timer", cache_dir=cache_dir, uri=uri)

def create_sundial_llm(cache_dir: Optional[str] = None,
                      uri: Optional[str] = None) -> IoTDBLLM:
    """创建Sundial模型的IoTDB LLM"""
    return IoTDBLLM("sundial", cache_dir=cache_dir, uri=uri)

def create_llm_from_uri(uri: str, 
                       cache_dir: Optional[str] = None) -> IoTDBLLM:
    """从URI创建IoTDB LLM"""
    return IoTDBLLM("custom", cache_dir=cache_dir, uri=uri)