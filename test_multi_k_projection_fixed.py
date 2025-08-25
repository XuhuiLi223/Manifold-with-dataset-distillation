import torch
import torch.nn as nn
import numpy as np
from multi_k_projection import MultiKProjectionNetwork, MultiKProjectionLoss, create_multi_k_projection_model
from model_dist import define_model
import argparse


class SimpleModel(nn.Module):
    """简单的特征提取器用于测试"""
    def __init__(self, input_channels, feature_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, feature_dim)
        
    def embed(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_multi_k_projection():
    """测试多K投影网络的基本功能"""
    print("=" * 50)
    print("Testing Multi-K Projection Network (Fixed)")
    print("=" * 50)
    
    # 设置参数
    batch_size = 8
    input_channels = 3
    img_size = 32
    feature_dim = 256
    num_classes = 10
    num_spaces = 10
    
    # 创建模拟数据
    real_images = torch.randn(batch_size, input_channels, img_size, img_size)
    syn_images = torch.randn(batch_size, input_channels, img_size, img_size)
    
    # 创建基础模型
    base_model = SimpleModel(input_channels, feature_dim)
    
    # 设置初始维度（从高到低）
    initial_dims = []
    for i in range(num_spaces):
        dim = int(feature_dim - i * (feature_dim - feature_dim//4) / (num_spaces - 1))
        initial_dims.append(dim)
    
    print(f"Initial dimensions: {initial_dims}")
    
    # 设置流形类型
    manifold_types = ['hyperbolic', 'spherical', 'euclidean', 'mixed']
    manifold_types = [manifold_types[i % len(manifold_types)] for i in range(num_spaces)]
    
    print(f"Manifold types: {manifold_types}")
    
    # 创建多K投影模型
    try:
        model = create_multi_k_projection_model(
            base_model=base_model,
            input_dim=feature_dim,
            num_classes=num_classes,
            num_spaces=num_spaces,
            initial_dims=initial_dims,
            manifold_types=manifold_types
        )
        print(f"\n✓ Model created successfully with {num_spaces} projection spaces")
    except Exception as e:
        print(f"\n✗ Error creating model: {e}")
        return
    
    # 测试前向传播
    print("\nTesting forward pass...")
    model.train()
    
    try:
        real_logits, real_info = model(real_images)
        syn_logits, syn_info = model(syn_images)
        
        print(f"✓ Forward pass successful")
        print(f"  Output logits shape: {real_logits.shape}")
        print(f"  Final features shape: {real_info['final_features'].shape}")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查每个空间的输出
    print("\nProjected features for each space:")
    for i, feat in enumerate(real_info['projected_features']):
        print(f"  Space {i}: shape={feat.shape}, manifold={manifold_types[i]}")
    
    # 检查空间权重
    print(f"\nSpace weights (first sample): {real_info['space_weights'][0].detach().numpy()}")
    print(f"Space utilization: {real_info['space_utilization'].numpy()}")
    
    # 检查曲率
    print(f"\nCurvatures: {real_info['curvatures'].detach().numpy()}")
    
    # 测试损失函数
    print("\nTesting loss function...")
    criterion = MultiKProjectionLoss(num_spaces=num_spaces)
    
    try:
        loss, loss_dict = criterion(real_info, syn_info)
        
        print(f"✓ Loss computation successful")
        print(f"  Total loss: {loss.item():.4f}")
        print("\n  Detailed losses:")
        for key, value in sorted(loss_dict.items()):
            print(f"    {key}: {value:.4f}")
    except Exception as e:
        print(f"✗ Error computing loss: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试反向传播
    print("\nTesting backward pass...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    try:
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        has_grad = False
        grad_info = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().sum().item()
                if grad_norm > 0:
                    has_grad = True
                    grad_info.append((name, grad_norm))
        
        if has_grad:
            print("✓ Gradients computed successfully")
            print(f"  Parameters with non-zero gradients: {len(grad_info)}")
            # 显示前5个有梯度的参数
            for name, norm in grad_info[:5]:
                print(f"    {name}: grad_norm={norm:.6f}")
        else:
            print("✗ No gradients found")
        
        optimizer.step()
        print("✓ Optimizer step completed")
    except Exception as e:
        print(f"✗ Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试维度降维
    print("\nTesting dimension reduction...")
    for i, proj in enumerate(model.projections[:3]):  # 只测试前3个
        if hasattr(proj, 'dim_reducer') and hasattr(proj.dim_reducer, 'dimension_weights'):
            weights = proj.dim_reducer.dimension_weights
            print(f"  Space {i} dimension weights stats: "
                  f"mean={weights.mean().item():.3f}, "
                  f"std={weights.std().item():.3f}, "
                  f"min={weights.min().item():.3f}, "
                  f"max={weights.max().item():.3f}")
    
    # 测试混合流形特性
    print("\nTesting mixed manifold properties...")
    mixed_indices = [i for i, t in enumerate(manifold_types) if t == 'mixed']
    if mixed_indices:
        for idx in mixed_indices[:2]:  # 测试前2个混合流形
            proj = model.projections[idx]
            if hasattr(proj, 'mix_weight'):
                weight = torch.sigmoid(proj.mix_weight).item()
                print(f"  Space {idx} (mixed): hyperbolic weight={weight:.3f}, spherical weight={1-weight:.3f}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 50)
    print("Testing Edge Cases")
    print("=" * 50)
    
    # 测试单个空间
    print("\nTest 1: Single projection space")
    try:
        base_model = SimpleModel(3, 64)
        model = create_multi_k_projection_model(
            base_model=base_model,
            input_dim=64,
            num_classes=5,
            num_spaces=1
        )
        x = torch.randn(2, 3, 16, 16)
        logits, info = model(x)
        print(f"✓ Single space test passed. Output shape: {logits.shape}")
    except Exception as e:
        print(f"✗ Single space test failed: {e}")
    
    # 测试大量空间
    print("\nTest 2: Many projection spaces (20)")
    try:
        base_model = SimpleModel(3, 128)
        model = create_multi_k_projection_model(
            base_model=base_model,
            input_dim=128,
            num_classes=10,
            num_spaces=20
        )
        x = torch.randn(2, 3, 16, 16)
        logits, info = model(x)
        print(f"✓ Many spaces test passed. Output shape: {logits.shape}")
        print(f"  Space utilization std: {info['space_utilization'].std().item():.3f}")
    except Exception as e:
        print(f"✗ Many spaces test failed: {e}")
    
    # 测试小维度
    print("\nTest 3: Small feature dimension")
    try:
        base_model = SimpleModel(1, 32)
        model = create_multi_k_projection_model(
            base_model=base_model,
            input_dim=32,
            num_classes=2,
            num_spaces=4
        )
        x = torch.randn(4, 1, 8, 8)
        logits, info = model(x)
        print(f"✓ Small dimension test passed. Output shape: {logits.shape}")
    except Exception as e:
        print(f"✗ Small dimension test failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Multi-K Projection')
    parser.add_argument('--test_edge_cases', action='store_true', 
                        help='Run edge case tests')
    
    args = parser.parse_args()
    
    # 运行主测试
    test_multi_k_projection()
    
    # 如果指定，运行边界测试
    if args.test_edge_cases:
        test_edge_cases()