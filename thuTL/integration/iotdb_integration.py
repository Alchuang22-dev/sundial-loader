import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from ..engine.llm_engine import LLM
from ..models.base_model import SequenceModel

class IoTDBModelAdapter:
    """IoTDB 模型适配器，将 thuTL 模型包装为 IoTDB 可调用的格式"""
    
    def __init__(self, model_id: str, model_type: str, config_path: Optional[str] = None):
        """
        初始化 IoTDB 模型适配器
        
        Args:
            model_id: 模型ID（如 '_timerxl', '_sundial'）
            model_type: 模型类型（如 'BUILT_IN_FORECAST'）
            config_path: 可选的配置文件路径
        """
        self.model_id = model_id
        self.model_type = model_type
        self.config_path = config_path
        
        # 从 model_id 推断实际模型名称
        self.actual_model = self._parse_model_name(model_id)
        
        # 初始化 LLM 引擎
        self.llm = None
        self._initialize_model()
        
        # 模型元信息
        self.metadata = self._get_model_metadata()
    
    def _parse_model_name(self, model_id: str) -> str:
        """从 IoTDB 模型ID 解析实际模型名称"""
        mapping = {
            '_timerxl': 'timer',
            '_sundial': 'sundial',
            '_timer': 'timer'
        }
        return mapping.get(model_id.lower(), model_id.lstrip('_'))
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            if self.config_path and Path(self.config_path).exists():
                self.llm = LLM(model=self.config_path, task="generate")
            else:
                self.llm = LLM(model=self.actual_model, task="generate")
            print(f"模型 {self.model_id} 初始化成功")
        except Exception as e:
            print(f"模型 {self.model_id} 初始化失败: {e}")
            raise
    
    def _get_model_metadata(self) -> Dict[str, Any]:
        """获取模型元信息"""
        metadata = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "state": "ACTIVE",
            "configs": {},
            "notes": f"Built-in {self.actual_model.upper()} model in IoTDB"
        }
        
        if self.llm:
            def extract_config(model):
                return {
                    "input_length": model.config.input_token_len,
                    "output_length": model.config.output_token_lens[0],
                    "hidden_size": model.config.hidden_size,
                    "num_layers": model.config.num_hidden_layers,
                    "model_type": model.config.model_type
                }
            
            metadata["configs"] = self.llm.apply_model(extract_config)
        
        return metadata
    
    def inference(self, 
                 sql_query: str,
                 generate_time: bool = True,
                 window: str = "tail(1024)",
                 output_length: Optional[int] = None,
                 keep_input: bool = False) -> pd.DataFrame:
        """
        执行推理，兼容 IoTDB 的 SQL 调用格式
        
        Args:
            sql_query: SQL 查询语句
            generate_time: 是否生成时间戳
            window: 数据窗口设置
            output_length: 输出长度
            keep_input: 是否保留输入数据
            
        Returns:
            包含预测结果的 DataFrame
        """
        # 1. 解析 SQL 并获取数据（这里需要实际的 IoTDB 连接）
        input_data = self._parse_sql_and_get_data(sql_query, window)
        
        # 2. 执行模型推理
        predictions = self._execute_inference(input_data, output_length)
        
        # 3. 生成结果 DataFrame
        result_df = self._format_results(
            input_data, predictions, generate_time, keep_input
        )
        
        return result_df
    
    def _parse_sql_and_get_data(self, sql_query: str, window: str) -> torch.Tensor:
        """解析 SQL 并获取数据（模拟实现）"""
        # 这里应该连接实际的 IoTDB 数据库
        # 目前提供模拟数据
        print(f"执行查询: {sql_query}")
        print(f"窗口设置: {window}")
        
        # 模拟时间序列数据
        if "tail(1024)" in window:
            data_length = 1024
        elif "tail(96)" in window:
            data_length = 96
        else:
            data_length = 100
        
        # 生成模拟数据
        np.random.seed(42)
        simulated_data = np.sin(np.linspace(0, 10, data_length)) + np.random.normal(0, 0.1, data_length)
        
        return torch.tensor(simulated_data, dtype=torch.float32)
    
    def _execute_inference(self, input_data: torch.Tensor, output_length: Optional[int]) -> torch.Tensor:
        """执行模型推理"""
        if self.llm is None:
            raise RuntimeError(f"模型 {self.model_id} 未正确初始化")
        
        # 根据模型调整输入长度
        def get_input_length(model):
            return model.config.input_token_len
        
        required_length = self.llm.apply_model(get_input_length)
        
        # 截取或填充输入数据
        if len(input_data) > required_length:
            input_data = input_data[-required_length:]
        elif len(input_data) < required_length:
            pad_length = required_length - len(input_data)
            input_data = torch.cat([
                torch.zeros(pad_length), input_data
            ])
        
        print(f"输入数据长度: {len(input_data)}")
        
        # 执行推理
        with torch.no_grad():
            predictions = self.llm.generate(input_data)
        
        # 如果指定了输出长度，则截取
        if output_length and len(predictions) > output_length:
            predictions = predictions[:output_length]
        
        print(f"预测数据长度: {len(predictions)}")
        return predictions
    
    def _format_results(self, 
                       input_data: torch.Tensor, 
                       predictions: torch.Tensor,
                       generate_time: bool,
                       keep_input: bool) -> pd.DataFrame:
        """格式化结果为 DataFrame"""
        
        # 生成时间戳
        if generate_time:
            base_time = datetime.now()
            if keep_input:
                # 包含输入数据的时间戳
                input_times = [
                    base_time + timedelta(seconds=i) 
                    for i in range(-len(input_data), 0)
                ]
                pred_times = [
                    base_time + timedelta(seconds=i) 
                    for i in range(len(predictions))
                ]
                all_times = input_times + pred_times
                all_values = torch.cat([input_data, predictions]).numpy()
            else:
                # 只有预测数据的时间戳
                all_times = [
                    base_time + timedelta(seconds=i) 
                    for i in range(len(predictions))
                ]
                all_values = predictions.numpy()
        else:
            all_times = list(range(len(predictions)))
            all_values = predictions.numpy()
        
        # 创建 DataFrame
        if generate_time:
            df = pd.DataFrame({
                'Time': all_times,
                'output0': all_values
            })
        else:
            df = pd.DataFrame({
                'time': all_times,
                'current': all_values
            })
        
        return df

class IoTDBInferenceEngine:
    """IoTDB 推理引擎，管理所有注册的模型"""
    
    def __init__(self):
        self.registered_models: Dict[str, IoTDBModelAdapter] = {}
        self._register_builtin_models()
    
    def _register_builtin_models(self):
        """注册内置模型"""
        builtin_models = [
            ("_timerxl", "BUILT_IN_FORECAST"),
            ("_sundial", "BUILT_IN_FORECAST"),
        ]
        
        for model_id, model_type in builtin_models:
            try:
                adapter = IoTDBModelAdapter(model_id, model_type)
                self.registered_models[model_id] = adapter
                print(f"注册模型: {model_id}")
            except Exception as e:
                print(f"注册模型失败 {model_id}: {e}")
    
    def show_models(self) -> pd.DataFrame:
        """显示所有注册的模型，对应 'show models' SQL 命令"""
        models_info = []
        
        for model_id, adapter in self.registered_models.items():
            metadata = adapter.metadata
            models_info.append({
                "ModelId": metadata["model_id"],
                "ModelType": metadata["model_type"],
                "State": metadata["state"],
                "Configs": str(metadata["configs"]) if metadata["configs"] else "",
                "Notes": metadata["notes"]
            })
        
        # 添加其他内置模型
        other_models = [
            ("_STLForecaster", "BUILT_IN_FORECAST", "ACTIVE", "", "Built-in model in IoTDB"),
            ("_NaiveForecaster", "BUILT_IN_FORECAST", "ACTIVE", "", "Built-in model in IoTDB"),
            ("_ARIMA", "BUILT_IN_FORECAST", "ACTIVE", "", "Built-in model in IoTDB"),
            ("_ExponentialSmoothing", "BUILT_IN_FORECAST", "ACTIVE", "", "Built-in model in IoTDB"),
            ("_GaussianHMM", "BUILT_IN_ANOMALY_DETECTION", "ACTIVE", "", "Built-in model in IoTDB"),
            ("_GMMHMM", "BUILT_IN_ANOMALY_DETECTION", "ACTIVE", "", "Built-in model in IoTDB"),
            ("_Stray", "BUILT_IN_ANOMALY_DETECTION", "ACTIVE", "", "Built-in model in IoTDB"),
        ]
        
        for model_info in other_models:
            models_info.append({
                "ModelId": model_info[0],
                "ModelType": model_info[1],
                "State": model_info[2],
                "Configs": model_info[3],
                "Notes": model_info[4]
            })
        
        return pd.DataFrame(models_info)
    
    def call_inference(self, 
                      model_id: str,
                      sql_query: str,
                      **kwargs) -> pd.DataFrame:
        """执行推理调用，对应 'call inference' SQL 命令"""
        if model_id not in self.registered_models:
            raise ValueError(f"模型 {model_id} 未注册或不支持")
        
        adapter = self.registered_models[model_id]
        return adapter.inference(sql_query, **kwargs)
    
    def forecast(self,
                model_id: str,
                input_query: str,
                keep_input: bool = False,
                output_length: int = 96,
                **kwargs) -> pd.DataFrame:
        """执行预测，对应 'forecast' SQL 函数"""
        if model_id not in self.registered_models:
            raise ValueError(f"模型 {model_id} 未注册或不支持")
        
        adapter = self.registered_models[model_id]
        return adapter.inference(
            input_query,
            generate_time=True,
            output_length=output_length,
            keep_input=keep_input,
            **kwargs
        )