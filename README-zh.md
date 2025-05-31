# Time Series Model loader
时序模型加载器框架 (正在开发中)

## 环境
```bash
pip install torch numpy matplotlib safetensors huggingface_hub
```

## 使用
```python
from thuTL import LLM
import torch

# Initialize model
llm = LLM(model="sundial", task="generate")

# Generate time series data
time_series = torch.randn(16)  # Input sequence

# Make predictions
result = llm.generate(time_series)
print(f"Prediction: {result}")
```
或者在根目录下直接运行`python example.py`