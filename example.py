"""
thuTL 包使用示例
演示如何使用 TimerXL 和 Sundial 模型进行时间序列预测
"""

import os
# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from thuTL import LLM

def generate_sample_data():
    """生成示例时间序列数据"""
    # 生成带有趋势和季节性的时间序列
    time = np.linspace(0, 10, 200)
    trend = 0.1 * time
    seasonal = 2 * np.sin(2 * np.pi * time) + np.sin(4 * np.pi * time)
    noise = np.random.normal(0, 0.2, len(time))
    series = trend + seasonal + noise
    return torch.tensor(series, dtype=torch.float32)

def test_timer_model():
    """测试 TimerXL 模型"""
    print("=" * 50)
    print("测试 TimerXL 模型")
    print("=" * 50)
    
    try:
        print("正在加载 TimerXL 模型...")
        timer_llm = LLM(model="timer", task="generate")
        
        # 打印模型配置信息
        timer_llm.apply_model(lambda model: print(f"模型配置: hidden_size={model.config.hidden_size}, num_heads={model.config.num_attention_heads}, head_dim={model.config.hidden_size//model.config.num_attention_heads}"))
        
        test_data = generate_sample_data()
        input_sequence = test_data[:96]
        
        print(f"输入序列长度: {len(input_sequence)}")
        print(f"输入数据范围: [{input_sequence.min():.3f}, {input_sequence.max():.3f}]")
        
        print("正在生成预测...")
        with torch.no_grad():
            predictions = timer_llm.generate(input_sequence)
        
        print(f"预测序列长度: {len(predictions)}")
        print(f"预测结果前5个值: {predictions[:5].tolist()}")
        
        return input_sequence, predictions
        
    except Exception as e:
        import traceback
        print(f"TimerXL 测试失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return None, None

def test_sundial_model():
    """测试 Sundial 模型"""
    print("\n" + "=" * 50)
    print("测试 Sundial 模型")
    print("=" * 50)
    
    try:
        # 初始化模型
        print("正在加载 Sundial 模型...")
        sundial_llm = LLM(model="sundial", task="generate")
        
        # 生成测试数据
        test_data = generate_sample_data()
        input_sequence = test_data[:16]  # Sundial 使用16个输入token
        
        print(f"输入序列长度: {len(input_sequence)}")
        print(f"输入数据范围: [{input_sequence.min():.3f}, {input_sequence.max():.3f}]")
        
        # 进行预测
        print("正在生成预测...")
        with torch.no_grad():
            predictions = sundial_llm.generate(input_sequence)
        
        print(f"预测序列长度: {len(predictions)}")
        print(f"预测结果前5个值: {predictions[:5].tolist()}")
        
        # 查看模型信息
        sundial_llm.apply_model(lambda model: print(f"模型类型: {type(model).__name__}"))
        sundial_llm.apply_model(lambda model: print(f"扩散步数: {model.config.num_sampling_steps}"))
        
        return input_sequence, predictions
        
    except Exception as e:
        print(f"Sundial 测试失败: {e}")
        return None, None

def test_local_model():
    """测试从本地路径加载模型"""
    print("\n" + "=" * 50)
    print("测试本地模型加载")
    print("=" * 50)
    
    try:
        # 假设本地有模型文件
        local_path = "./models/sundial"  # 本地模型路径
        print(f"尝试从本地路径加载: {local_path}")
        
        local_llm = LLM(model=local_path, task="generate")
        
        # 测试数据
        test_data = torch.randn(16)  # 随机测试数据
        
        print("正在使用本地模型生成预测...")
        with torch.no_grad():
            predictions = local_llm.generate(test_data)
        
        print(f"本地模型预测完成，输出长度: {len(predictions)}")
        
    except Exception as e:
        print(f"本地模型测试失败: {e}")
        print("这是正常的，如果你还没有下载模型到本地")

def visualize_results(timer_input, timer_pred, sundial_input, sundial_pred):
    """可视化预测结果"""
    print("\n" + "=" * 50)
    print("可视化预测结果")
    print("=" * 50)
    
    try:
        plt.figure(figsize=(15, 10))
        
        # TimerXL 结果
        if timer_input is not None and timer_pred is not None:
            plt.subplot(2, 2, 1)
            plt.plot(timer_input.numpy(), label='Timer Input', color='blue')
            plt.title('TimerXL - 输入序列')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(timer_pred.numpy(), label='Timer Prediction', color='red')
            plt.title('TimerXL - 预测结果')
            plt.legend()
            plt.grid(True)
        
        # Sundial 结果
        if sundial_input is not None and sundial_pred is not None:
            plt.subplot(2, 2, 3)
            plt.plot(sundial_input.numpy(), label='Sundial Input', color='green')
            plt.title('Sundial - 输入序列')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.plot(sundial_pred.numpy(), label='Sundial Prediction', color='orange')
            plt.title('Sundial - 预测结果')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("结果已保存到 prediction_results.png")
        
    except Exception as e:
        print(f"可视化失败: {e}")
        print("请安装 matplotlib: pip install matplotlib")

def advanced_usage_examples():
    """高级使用示例"""
    print("\n" + "=" * 50)
    print("高级使用示例")
    print("=" * 50)
    
    try:
        # 批量处理
        print("1. 批量预测示例")
        llm = LLM(model="sundial", task="generate")
        
        # 生成多个时间序列
        batch_data = [generate_sample_data()[:16] for _ in range(3)]
        batch_predictions = []
        
        for i, series in enumerate(batch_data):
            pred = llm.generate(series)
            batch_predictions.append(pred)
            print(f"  序列 {i+1} 预测完成，输出长度: {len(pred)}")
        
        # 模型参数查看
        print("\n2. 模型参数信息")
        def print_model_info(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  总参数量: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        llm.apply_model(print_model_info)
        
        # 配置信息查看
        print("\n3. 配置信息")
        def print_config(model):
            config = model.config
            print(f"  模型类型: {config.model_type}")
            print(f"  隐藏层大小: {config.hidden_size}")
            print(f"  注意力头数: {config.num_attention_heads}")
            print(f"  输入长度: {config.input_token_len}")
            print(f"  输出长度: {config.output_token_lens}")
        
        llm.apply_model(print_config)
        
    except Exception as e:
        print(f"高级示例失败: {e}")

def main():
    """主测试函数"""
    print("thuTL 时间序列大模型库测试")
    print("Author: Zeyu Zhang")
    print("Version: 1.0")
    
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 基础测试
    timer_input, timer_pred = test_timer_model()
    sundial_input, sundial_pred = test_sundial_model()
    
    # 本地模型测试
    test_local_model()
    
    # 可视化结果
    visualize_results(timer_input, timer_pred, sundial_input, sundial_pred)
    
    # 高级用法示例
    advanced_usage_examples()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("如需更多功能，请查看文档或源代码")
    print("=" * 50)

if __name__ == "__main__":
    main()