import yaml
import json
from pathlib import Path
from typing import Dict, Any

# class IoTDBDeploymentConfig:
#     """IoTDB 部署配置管理"""
    
#     def __init__(self, config_path: Optional[str] = None):
#         self.config_path = config_path or "iotdb_config.yaml"
#         self.config = self._load_config()
    
#     def _load_config(self) -> Dict[str, Any]:
#         """加载配置文件"""
#         config_file = Path(self.config_path)
        
#         if config_file.exists():
#             with open(config_file, 'r', encoding='utf-8') as f:
#                 return yaml.safe_load(f)
#         else:
#             # 默认配置
#             default_config = {
#                 "iotdb": {
#                     "host": "localhost",
#                     "port": 6667,
#                     "username": "root",
#                     "password": "root"
#                 },
#                 "models": {
#                     "cache_dir": "./models",
#                     "hf_endpoint": "https://hf-mirror.com",
#                     "use_proxy": False,
#                     "proxy_port": 7890
#                 },
#                 "inference": {
#                     "batch_size": 1,
#                     "device": "cpu",
#                     "max_sequence_length": 1024
#                 }
#             }
#             self._save_config(default_config)
#             return default_config
    
#     def _save_config(self, config: Dict[str, Any]):
#         """保存配置文件"""
#         with open(self.config_path, 'w', encoding='utf-8') as f:
#             yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
#     def get_iotdb_config(self) -> Dict[str, Any]:
#         """获取 IoTDB 连接配置"""
#         return self.config.get("iotdb", {})
    
#     def get_model_config(self) -> Dict[str, Any]:
#         """获取模型配置"""
#         return self.config.get("models", {})
    
#     def get_inference_config(self) -> Dict[str, Any]:
#         """获取推理配置"""
#         return self.config.get("inference", {})