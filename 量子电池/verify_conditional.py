"""
验证脚本：测试条件参数集成是否正确
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.dataset import QuantumBatteryDataset
from src.model.lstm import build_model

def test_dataset():
    """测试 Dataset 返回条件参数"""
    print("=" * 60)
    print("测试 1: Dataset 返回格式")
    print("=" * 60)
    
    data_dir = Path("experiments/output/a_n3/data")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    try:
        dataset = QuantumBatteryDataset(data_dir, window_size=20, train=True)
        sample = dataset[0]
        
        print(f"✓ Dataset 加载成功")
        print(f"  样本数量: {len(dataset)}")
        
        if len(sample) == 3:
            X, y, cond = sample
            print(f"✓ 返回 3 个元素")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  cond shape: {cond.shape}")
            print(f"  cond values: {cond.numpy()}")
            return True
        else:
            print(f"❌ 返回 {len(sample)} 个元素，期望 3 个")
            return False
            
    except Exception as e:
        print(f"❌ Dataset 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """测试模型接收条件参数"""
    print("\n" + "=" * 60)
    print("测试 2: 模型条件参数集成")
    print("=" * 60)
    
    try:
        # 构建带条件参数的模型
        model = build_model(
            input_dim=128,
            output_dim=128,
            lstm_layers=[128, 64],
            dense_units=32,
            dropout=0.1,
            cond_dim=4  # Ω, γ, τ_m, Ω/γ
        )
        
        print(f"✓ 模型构建成功 (cond_dim=4)")
        
        # 测试前向传播
        batch_size = 4
        window_size = 20
        X = torch.randn(batch_size, window_size, 128)
        cond = torch.rand(batch_size, 4)
        
        output = model(X, cond=cond)
        
        print(f"✓ 前向传播成功")
        print(f"  输入 X: {X.shape}")
        print(f"  输入 cond: {cond.shape}")
        print(f"  输出: {output.shape}")
        
        # 测试不带条件参数
        output_no_cond = model(X)
        print(f"✓ 不带条件参数也能工作: {output_no_cond.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_loop():
    """测试训练循环"""
    print("\n" + "=" * 60)
    print("测试 3: 训练循环兼容性")
    print("=" * 60)
    
    try:
        from torch.utils.data import DataLoader
        from src.training.train import train_epoch, validate_epoch
        from src.model.loss import PhysicsInformedLoss
        
        data_dir = Path("experiments/output/a_n3/data")
        if not data_dir.exists():
            print(f"⚠ 跳过训练循环测试（数据不存在）")
            return True
        
        dataset = QuantumBatteryDataset(data_dir, window_size=20, train=True)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        model = build_model(
            input_dim=128,
            output_dim=128,
            lstm_layers=[128, 64],
            dense_units=32,
            dropout=0.1,
            cond_dim=4
        )
        
        criterion = PhysicsInformedLoss(
            lambda_trace=1.0,
            lambda_herm=1.0,
            lambda_pos=0.1,
            d=8  # N=3 TLS → d=2^3=8
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device('cpu')
        
        model.to(device)
        
        # 测试一个 epoch
        metrics = train_epoch(
            model, loader, criterion, optimizer, device,
            epoch=1, total_epochs=1
        )
        
        print(f"✓ 训练循环成功")
        print(f"  训练损失: {metrics['train_loss']:.6f}")
        
        # 测试验证循环
        val_metrics = validate_epoch(
            model, loader, criterion, device,
            epoch=1, total_epochs=1
        )
        
        print(f"✓ 验证循环成功")
        print(f"  验证损失: {val_metrics['val_loss']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练循环测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("条件参数集成验证")
    print("=" * 60)
    
    results = []
    
    # 测试 1: Dataset
    results.append(("Dataset", test_dataset()))
    
    # 测试 2: Model
    results.append(("Model", test_model()))
    
    # 测试 3: Training Loop
    results.append(("Training Loop", test_training_loop()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
