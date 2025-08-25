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
    print("Testing Multi-K Projection Network")
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
    model = create_multi_k_projection_model(
        base_model=base_model,
        input_dim=feature_dim,
        num_classes=num_classes,
        num_spaces=num_spaces,
        initial_dims=initial_dims,
        manifold_types=manifold_types
    )
    
    print(f"\nModel created with {num_spaces} projection spaces")
    
    # 测试前向传播
    print("\nTesting forward pass...")
    model.train()
    
    real_logits, real_info = model(real_images)
    syn_logits, syn_info = model(syn_images)
    
    print(f"Output logits shape: {real_logits.shape}")
    print(f"Final features shape: {real_info['final_features'].shape}")
    
    # 检查每个空间的输出
    print("\nProjected features for each space:")
    for i, feat in enumerate(real_info['projected_features']):
        print(f"  Space {i}: shape={feat.shape}, manifold={manifold_types[i]}")
    
    # 检查空间权重
    print(f"\nSpace weights: {real_info['space_weights'][0].detach().numpy()}")
    print(f"Space utilization: {real_info['space_utilization'].numpy()}")
    
    # 检查曲率
    print(f"\nCurvatures: {real_info['curvatures'].detach().numpy()}")
    
    # 测试损失函数
    print("\nTesting loss function...")
    criterion = MultiKProjectionLoss(num_spaces=num_spaces)
    
    loss, loss_dict = criterion(real_info, syn_info)
    
    print(f"Total loss: {loss.item():.4f}")
    print("\nDetailed losses:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试反向传播
    print("\nTesting backward pass...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    if has_grad:
        print("✓ Gradients computed successfully")
    else:
        print("✗ No gradients found")
    
    optimizer.step()
    
    # 测试维度降维
    print("\nTesting dimension reduction...")
    for i, proj in enumerate(model.projections[:3]):  # 只测试前3个
        if hasattr(proj, 'dim_reducer'):
            weights = proj.dim_reducer.dimension_weights
            print(f"  Space {i} dimension weights stats: "
                  f"mean={weights.mean().item():.3f}, "
                  f"std={weights.std().item():.3f}, "
                  f"min={weights.min().item():.3f}, "
                  f"max={weights.max().item():.3f}")
    
    print("\n" + "=" * 50)
    print("Multi-K Projection test completed successfully!")
    print("=" * 50)


def test_with_real_model(args):
    """使用真实模型测试"""
    print("\n" + "=" * 50)
    print("Testing with real model architecture")
    print("=" * 50)
    
    # 设置参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    img_size = 32 if args.dataset in ['cifar10', 'cifar100'] else 224
    num_channels = 3 if args.dataset != 'mnist' else 1
    num_classes = 10 if args.dataset in ['cifar10', 'mnist', 'fashion', 'svhn'] else 100
    num_spaces = args.num_spaces
    
    print(f"Dataset: {args.dataset}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Number of channels: {num_channels}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of spaces: {num_spaces}")
    print(f"Device: {device}")
    
    # 创建模拟数据
    real_images = torch.randn(batch_size, num_channels, img_size, img_size).to(device)
    syn_images = torch.randn(batch_size, num_channels, img_size, img_size).to(device)
    
    # 创建基础模型
    base_model = define_model(args, num_classes).to(device)
    
    # 获取特征维度
    with torch.no_grad():
        dummy_features = base_model.embed(real_images[:1])
        feature_dim = dummy_features.shape[1]
    
    print(f"Feature dimension: {feature_dim}")
    
    # 创建多K投影模型
    model = create_multi_k_projection_model(
        base_model=base_model,
        input_dim=feature_dim,
        num_classes=num_classes,
        num_spaces=num_spaces
    ).to(device)
    
    # 测试前向传播
    print("\nTesting forward pass with real model...")
    model.train()
    
    real_logits, real_info = model(real_images)
    syn_logits, syn_info = model(syn_images)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {real_logits.shape}")
    
    # 测试损失计算
    criterion = MultiKProjectionLoss(num_spaces=num_spaces)
    loss, loss_dict = criterion(real_info, syn_info)
    
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {loss.item():.4f}")
    
    # 测试内存使用
    if device == 'cuda':
        print(f"\nMemory usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Multi-K Projection')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'mnist', 'fashion', 'svhn', 'imagenet'])
    parser.add_argument('--net_type', type=str, default='convnet')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--num_spaces', type=int, default=10)
    parser.add_argument('--test_real', action='store_true', help='Test with real model')
    
    args = parser.parse_args()
    
    # 运行基础测试
    test_multi_k_projection()
    
    # 如果指定，运行真实模型测试
    if args.test_real:
        test_with_real_model(args)