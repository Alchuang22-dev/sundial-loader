import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import json

class ModelDownloader:
    def __init__(self, cache_dir="./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_model_files(self, repo_id, model_name):
        """下载指定模型的config和权重文件"""
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        files_to_download = [
            "config.json",
            "model.safetensors"
        ]
        
        downloaded_files = {}
        
        for filename in files_to_download:
            try:
                print(f"Downloading {filename} in {model_name}...")
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(model_dir),
                    local_files_only=False
                )
                downloaded_files[filename] = file_path
                print(f"✓ {filename} downloaded")
                
            except Exception as e:
                print(f"✗ [ERROR] download {filename} failed: {e}")
                
        return downloaded_files
    
    def download_timerxl(self):
        """下载TimerXL模型文件"""
        return self.download_model_files(
            repo_id="thuml/timer-base-84m", 
            model_name="timerxl"
        )
    
    def download_sundial(self):
        """下载sundial模型文件"""
        return self.download_model_files(
            repo_id="thuml/sundial-base-128m", 
            model_name="sundial"
        )

# 使用示例
if __name__ == "__main__":
    downloader = ModelDownloader()
    
    # 下载TimerXL
    timerxl_files = downloader.download_timerxl()
    
    # 下载sundial
    sundial_files = downloader.download_sundial()
    
    print("Downloaded files:")
    print("TimerXL:", timerxl_files)
    print("Sundial:", sundial_files)