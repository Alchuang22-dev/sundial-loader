# Time Series Model loader
> 2025/06/04
TimerXL and Sundial Loader for IoTDB (under development)

the loader could be imported as `thuTL`
A PyTorch-based library for time series forecasting using large language models, designed to be compatible with vLLM-style APIs while supporting advanced time series models like TimerXL and Sundial.

## Features
+ vLLM-compatible API: Easy-to-use interface similar to vLLM
+ Multiple Model Support: TimerXL and Sundial models with their specific architectures
+ Automatic Model Download: Seamless integration with HuggingFace Hub
+ Safetensors Support: Secure model weight loading
+ Flexible Configuration: Support for different model configurations
+ Time-aware Attention: RoPE (Rotary Position Embedding) for temporal modeling

## Installation
```bash
pip install torch numpy matplotlib safetensors huggingface_hub
```
## Quick Start
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
## Project Structure
```angelscript
thuTL/
├── __init__.py
├── config/    # discarded
├── models/
│   ├── __init__.py
│   ├── base_model.py        # Base model class
│   ├── attention.py         # Attention mechanisms
│   ├── timerxl/          # TimerXL implementation
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── configuration.py
│   │   └── modeling_timerxl.py
│   └── sundial/          # Sundial implementation
├── engine/
│   ├── __init__.py
│   └── llm_engine.py       # Main LLM engine
├── integration/            # Apache IoTDB ports
└── utils/
    ├── __init__.py
    └── model_loader.py      # Model download utilities
```
## Supported Models
+ TimerXL
    - Input Length: 96 tokens
    - Output Length: 96 tokens
    - Architecture: Time-aware Transformer with attention dropout
    - Key Features: RoPE positional encoding, multi-head attention
+ Sundial
    - Input Length: 16 tokens
    - Output Length: 720 tokens
    - Architecture: Diffusion-Flow hybrid with time-attention
    - Key Features: Diffusion sampling, flow transformations, time-step embeddings
## Configuration
The library automatically loads model configurations from config.json files:
The configurations would be stored in `models/model_name/configuration.py`
```python
# Base configuration shared by all models
@dataclass
class BaseSeqConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    input_token_len: int
    output_token_lens: List[int]
    # ... other common fields

# Model-specific configurations
@dataclass
class SundialConfig(BaseSeqConfig):
    diffusion_batch_mul: int = 4
    flow_loss_depth: int = 3
    num_sampling_steps: int = 50
    dropout_rate: float = 0.1

@dataclass
class TimerConfig(BaseSeqConfig):
    attention_dropout: float = 0.0
```
## Usage Examples
### Basic Prediction
```python
from thuTL import LLM
import torch

# Load TimerXL model
timer_llm = LLM(model="timer", task="generate")

# Create sample time series (96 points for TimerXL)
sample_data = torch.sin(torch.linspace(0, 4*3.14159, 96))

# Generate prediction
prediction = timer_llm.generate(sample_data)
print(f"Input length: {len(sample_data)}")
print(f"Prediction length: {len(prediction)}")
```
### Batch Processing
```python
# Process multiple time series
batch_data = [torch.randn(16) for _ in range(5)]  # For Sundial
sundial_llm = LLM(model="sundial", task="generate")

predictions = []
for series in batch_data:
    pred = sundial_llm.generate(series)
    predictions.append(pred)
```
### Model Information
```python
# Get model details
llm.apply_model(lambda model: print(f"Model type: {type(model).__name__}"))
llm.apply_model(lambda model: print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}"))
```
### Local Model Loading
```python
# Load from local path
local_llm = LLM(model="./path/to/model", task="generate")
```
### Loading Model from Internet
```python
try:
        llm_with_uri = LLM(
            model="custom_model",
            task="generate",
            uri="https://example.com/models/timer"  
        )
    except Exception as e:

# or
try:
        llm_direct_uri = LLM(
            model="https://example.com/models/sundial",
            task="generate"
        )
    except Exception as e:
```
### Generate Function
```python
from thuTL import LLM, IoTDBLLM, create_timer_llm, create_sundial_llm, create_llm_from_uri

# IoTDBLLM will fix the task config
    try:
        iotdb_llm = IoTDBLLM(model="timer")
        
        sample_data = torch.randn(96)
        prediction = iotdb_llm.forecast(sample_data)    
    except Exception as e:
    
# or use "quick create" function
    try:
        # TimerXL
        timer_llm = create_timer_llm()
        timer_pred = timer_llm.forecast(torch.randn(96))
        
        # Sundial
        sundial_llm = create_sundial_llm()
        sundial_pred = sundial_llm.forecast(torch.randn(16))     
    except Exception as e:
```
## Model Architecture Details
### Attention Mechanism
The library implements time-aware attention with RoPE:

```python
class BaseTimeAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rope_theta: float = 10000.0):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.rope = RoPEEmbedding(self.head_dim, theta=rope_theta)
        # ... other components
```
### Diffusion-Flow (Sundial)
```python
class SundialBlock(nn.Module):
    def forward(self, x: torch.Tensor, timestep_emb: Optional[torch.Tensor] = None):
        # Time attention
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        
        # Flow transformation
        if timestep_emb is not None:
            x = x + timestep_emb.unsqueeze(1)
        flow_out = self.flow_layer(self.ln2(x))
        x = x + flow_out
        
        # Feed forward
        ff_out = self.feed_forward(self.ln3(x))
        return x + ff_out
```
## Troubleshooting
### Common Issues
+ Import Errors: Ensure all __init__.py files are present and use relative imports
+ Model Registration: Make sure model classes are imported before use
+ Tensor Dimension Mismatch: Check that RoPE dimensions match attention head dimensions
+ OpenMP Conflicts: Set os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" if needed
## Debug Mode
```python
# Enable debug output
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check registered models
from thuTL.models.base_model import SequenceModel
print(f"Registered models: {list(SequenceModel._registry.keys())}")
```
## Configuration Files
TimerXL Config Example
```json
{
  "model_type": "timer",
  "hidden_size": 1024,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "input_token_len": 96,
  "output_token_lens": [96],
  "attention_dropout": 0.0,
  "rope_theta": 10000
}
```
Sundial Config Example
```json
{
  "model_type": "sundial",
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "input_token_len": 16,
  "output_token_lens": [720],
  "diffusion_batch_mul": 4,
  "num_sampling_steps": 50,
  "dropout_rate": 0.1
}
```
## Integration
### Apache IoTDB Integration
The library is designed for seamless integration with Apache IoTDB and other time series databases:
```
# Example IoTDB integration
def predict_timeseries(session, device_path, prediction_length):
    # Fetch data from IoTDB
    data = session.execute_query(f"SELECT value FROM {device_path}")
    
    # Convert to tensor
    series = torch.tensor(data.values)
    
    # Generate prediction
    llm = LLM(model="sundial", task="generate")
    prediction = llm.generate(series)
    
    return prediction
```

## License
--

## Citation
--


