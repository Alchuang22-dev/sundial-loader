import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import json
import requests
from urllib.parse import urlparse

# 网络测试器
class ModelDownloader:
    def __init__(self, cache_dir="./models", hf_endpoint=None, proxy_port=None, use_proxy=False):
        """
        初始化模型下载器
        
        Args:
            cache_dir (str): 本地缓存目录
            hf_endpoint (str): HuggingFace镜像地址，默认为 https://hf-mirror.com
            proxy_port (int): 代理端口，默认为 7890
            use_proxy (bool): 是否使用代理，默认为 False
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 设置镜像地址
        self.hf_endpoint = hf_endpoint or "https://hf-mirror.com"
        
        # 设置代理配置
        self.proxy_port = proxy_port or 7890
        self.use_proxy = use_proxy
        
        # 配置环境变量和代理
        self._setup_environment()
    
    def _setup_environment(self):
        """设置环境变量和代理配置"""
        # 设置 HuggingFace 镜像环境变量
        os.environ["HF_ENDPOINT"] = self.hf_endpoint
        print(f"🌍 使用 HuggingFace 镜像: {self.hf_endpoint}")
        
        # 配置代理（如果启用）
        if self.use_proxy:
            proxy_url = f"http://127.0.0.1:{self.proxy_port}"
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
            print(f"🚀 使用代理: {proxy_url}")
        
        # 测试网络连接
        self._test_connection()
    
    def _test_connection(self):
        """测试网络连接"""
        try:
            # 测试镜像连接
            response = requests.get(self.hf_endpoint, timeout=10)
            if response.status_code == 200:
                print(f"✅ 镜像连接正常: {self.hf_endpoint}")
            else:
                print(f"⚠️  镜像连接异常: {self.hf_endpoint}, 状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ 网络连接测试失败: {e}")
            print("💡 建议检查网络连接或代理设置")
    
    def set_proxy(self, port=7890, enabled=True):
        """
        动态设置代理
        
        Args:
            port (int): 代理端口
            enabled (bool): 是否启用代理
        """
        self.proxy_port = port
        self.use_proxy = enabled
        
        if enabled:
            proxy_url = f"http://127.0.0.1:{port}"
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
            print(f"🚀 代理已启用: {proxy_url}")
        else:
            # 移除代理环境变量
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
            print("🔒 代理已禁用")
    
    def set_mirror(self, endpoint="https://hf-mirror.com"):
        """
        动态设置镜像地址
        
        Args:
            endpoint (str): 镜像地址
        """
        self.hf_endpoint = endpoint
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"🌍 镜像地址已更新: {endpoint}")
        self._test_connection()
    
    def download_model_files(self, repo_id, model_name, files=None):
        """
        下载指定模型的文件
        
        Args:
            repo_id (str): HuggingFace 仓库ID
            model_name (str): 本地模型名称
            files (list): 要下载的文件列表，默认为 ["config.json", "model.safetensors"]
        """
        if files is None:
            files = ["config.json", "model.safetensors"]
            
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        downloaded_files = {}
        
        print(f"📦 开始下载模型: {repo_id}")
        print(f"📂 保存路径: {model_dir}")
        
        for filename in files:
            try:
                print(f"⬇️  正在下载 {model_name} 的 {filename}...")
                
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(model_dir),
                    local_files_only=False,
                    resume_download=True  # 支持断点续传
                )
                
                downloaded_files[filename] = file_path
                
                # 显示文件大小
                file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                print(f"✅ {filename} 下载完成 ({file_size:.2f} MB)")
                
            except Exception as e:
                print(f"❌ 下载 {filename} 失败: {e}")
                print(f"💡 尝试检查网络连接或切换镜像")
                
        return downloaded_files
    
    def download_timerxl(self, files=None):
        """下载TimerXL模型文件"""
        return self.download_model_files(
            repo_id="thuml/timer-base-84m", 
            model_name="timerxl",
            files=files
        )
    
    def download_sundial(self, files=None):
        """下载Sundial模型文件"""
        return self.download_model_files(
            repo_id="thuml/sundial-base-128m", 
            model_name="sundial",
            files=files
        )
    
    def list_downloaded_models(self):
        """列出已下载的模型"""
        print("📋 已下载的模型:")
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                files = list(model_dir.glob("*"))
                print(f"  📁 {model_dir.name}: {len(files)} 个文件")
                for file in files:
                    file_size = file.stat().st_size / (1024 * 1024)
                    print(f"    📄 {file.name} ({file_size:.2f} MB)")
    
    def get_config(self):
        """获取当前配置信息"""
        config = {
            "cache_dir": str(self.cache_dir),
            "hf_endpoint": self.hf_endpoint,
            "proxy_port": self.proxy_port,
            "use_proxy": self.use_proxy,
            "environment_vars": {
                "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
                "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
                "HTTPS_PROXY": os.environ.get("HTTPS_PROXY")
            }
        }
        return config

# 使用示例
if __name__ == "__main__":
    # 方式1: 使用默认配置（HF-Mirror镜像，无代理）
    print("=== 默认配置 ===")
    downloader = ModelDownloader()
    
    # 方式2: 启用代理
    print("\n=== 启用代理配置 ===")
    downloader_with_proxy = ModelDownloader(
        cache_dir="./models", 
        use_proxy=True, 
        proxy_port=7890
    )
    
    # 方式3: 自定义镜像和代理
    print("\n=== 自定义配置 ===")
    custom_downloader = ModelDownloader(
        cache_dir="./models",
        hf_endpoint="https://hf-mirror.com",
        proxy_port=7890,
        use_proxy=True
    )
    
    # 动态修改配置
    print("\n=== 动态配置修改 ===")
    downloader.set_proxy(port=7890, enabled=True)  # 启用代理
    downloader.set_mirror("https://huggingface.co")  # 切换回官方镜像
    
    # 查看配置
    print("\n=== 当前配置 ===")
    config = downloader.get_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # 下载模型
    print("\n=== 开始下载 ===")
    try:
        # 下载TimerXL
        timerxl_files = downloader.download_timerxl()
        
        # 下载Sundial
        sundial_files = downloader.download_sundial()
        
        print("\n📋 下载完成的文件:")
        print("TimerXL:", timerxl_files)
        print("Sundial:", sundial_files)
        
        # 列出已下载的模型
        downloader.list_downloaded_models()
        
    except Exception as e:
        print(f"❌ 下载过程中出现错误: {e}")