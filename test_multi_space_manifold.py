#!/usr/bin/env python3
"""
多空间可学习流形模型测试脚本
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from learnable_manifold_multi_space import (
    MultiSpaceLearnableManifoldNetwork,
    MultiSpaceManifoldMatchingLoss,
    AdaptiveDimensionProjection,
    MultiSpaceManifoldProjection,
    create_multi_space_manifold_model
)
import models.convnet as CN


class DummyBaseModel(nn.Module):
    """用于测试的简单基础模型"""
    def __init__(self, input_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def embed(self, x):
        if len(x.shape) == 4:  # 如果是图像数据，flatten
            x = x.view(x.size(0), -1)
        return self.features(x)


def test_adaptive_dimension_projection():
    """测试自适应维度投影"""
    print("=== 测试自适应维度投影 ===")
    
    batch_size = 32
    input_dim = 256
    target_range = (64, 128)
    
    # 创建投影层
    proj = AdaptiveDimensionProjection(
        input_dim=input_dim,
        target_dim_range=target_range,
        init_target_dim=96
    )
    
    # 测试输入
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    projected, current_dim = proj(x)
    
    print(f"输入维度: {input_dim}")
    print(f"目标维度范围: {target_range}")
    print(f"当前目标维度: {current_dim}")
    print(f"投影后形状: {projected.shape}")
    print(f"维度logit: {proj.dim_logit.item():.4f}")
    
    # 测试维度变化
    print("\n测试维度参数变化:")
    for logit_val in [-2, -1, 0, 1, 2]:
        proj.dim_logit.data = torch.tensor(float(logit_val))
        _, dim = proj(x)
        print(f"  logit={logit_val:2d} -> dim={dim}")
    
    assert projected.shape[0] == batch_size
    assert target_range[0] <= current_dim <= target_range[1]
    print("✓ 自适应维度投影测试通过")


def test_multi_space_manifold_projection():
    """测试多空间流形投影"""
    print("\n=== 测试多空间流形投影 ===")
    
    batch_size = 16
    input_dim = 128
    target_range = (32, 64)
    
    # 测试不同的流形空间
    for space_id in range(5):
        print(f"\n测试空间 {space_id}:")
        proj = MultiSpaceManifoldProjection(
            input_dim=input_dim,
            space_id=space_id,
            target_dim_range=target_range
        )
        
        print(f"  流形类型: {proj.manifold_type}")
        print(f"  曲率: {proj.curvature if hasattr(proj, 'curvature') else 'N/A'}")
        
        x = torch.randn(batch_size, input_dim)
        feat, manifold, dim = proj.project_to_manifold(x)
        tangent = proj.to_tangent_space(feat, manifold)
        
        print(f"  投影后维度: {dim}")
        print(f"  流形特征形状: {feat.shape}")
        print(f"  切空间特征形状: {tangent.shape}")
        
        assert feat.shape[0] == batch_size
        assert tangent.shape[0] == batch_size
    
    print("✓ 多空间流形投影测试通过")


def test_multi_space_network():
    """测试完整的多空间网络"""
    print("\n=== 测试多空间网络 ===")
    
    batch_size = 8
    input_dim = 64
    num_classes = 10
    num_spaces = 6
    target_range = (16, 32)
    
    # 创建基础模型
    base_model = DummyBaseModel(input_dim)
    
    # 创建多空间网络
    model = create_multi_space_manifold_model(
        base_model=base_model,
        input_dim=input_dim,
        num_classes=num_classes,
        num_spaces=num_spaces,
        target_dim_range=target_range
    )
    
    print(f"创建了 {num_spaces} 个流形空间")
    print(f"维度范围: {target_range}")
    
    # 测试前向传播
    x = torch.randn(batch_size, input_dim)
    logits, info = model(x, epoch=0)
    
    print(f"输入形状: {x.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"当前维度: {info['current_dims']}")
    print(f"融合权重形状: {info['fusion_gates'].shape}")
    print(f"融合权重均值: {info['fusion_gates'].mean(dim=0)}")
    
    # 验证输出
    assert logits.shape == (batch_size, num_classes)
    assert len(info['manifold_features']) == num_spaces
    assert len(info['current_dims']) == num_spaces
    assert info['fusion_gates'].shape == (batch_size, num_spaces)
    
    print("✓ 多空间网络测试通过")


def test_loss_function():
    """测试损失函数"""
    print("\n=== 测试损失函数 ===")
    
    batch_size = 16
    num_spaces = 4
    feature_dim = 32
    
    # 创建模拟的真实数据和合成数据信息
    real_info = {
        'manifold_features': [torch.randn(batch_size, feature_dim) for _ in range(num_spaces)],
        'manifold_info': [
            {
                'features': torch.randn(batch_size, feature_dim),
                'manifold_type': ['euclidean', 'hyperbolic', 'spherical', 'hyperbolic'][i],
                'curvature': 1.0 + 0.5 * i
            } for i in range(num_spaces)
        ],
        'fused_features': torch.randn(batch_size, feature_dim),
        'fusion_gates': torch.softmax(torch.randn(batch_size, num_spaces), dim=1),
        'current_dims': [feature_dim] * num_spaces,
        'curvatures': [1.0 + 0.5 * i for i in range(num_spaces)]
    }
    
    syn_info = {
        'manifold_features': [torch.randn(batch_size, feature_dim) for _ in range(num_spaces)],
        'manifold_info': [
            {
                'features': torch.randn(batch_size, feature_dim),
                'manifold_type': ['euclidean', 'hyperbolic', 'spherical', 'hyperbolic'][i],
                'curvature': 1.2 + 0.4 * i
            } for i in range(num_spaces)
        ],
        'fused_features': torch.randn(batch_size, feature_dim),
        'fusion_gates': torch.softmax(torch.randn(batch_size, num_spaces), dim=1),
        'current_dims': [feature_dim] * num_spaces,
        'curvatures': [1.2 + 0.4 * i for i in range(num_spaces)]
    }
    
    # 创建损失函数
    criterion = MultiSpaceManifoldMatchingLoss(num_spaces=num_spaces)
    
    # 计算损失
    total_loss, loss_dict = criterion(None, None, real_info, syn_info)
    
    print(f"总损失: {total_loss.item():.4f}")
    print("损失分量:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() >= 0
    assert len(loss_dict) > 0
    
    print("✓ 损失函数测试通过")


def test_dimension_evolution():
    """测试维度演化"""
    print("\n=== 测试维度演化 ===")
    
    batch_size = 8
    input_dim = 128
    num_classes = 5
    num_spaces = 3
    target_range = (16, 64)
    
    base_model = DummyBaseModel(input_dim)
    model = create_multi_space_manifold_model(
        base_model=base_model,
        input_dim=input_dim,
        num_classes=num_classes,
        num_spaces=num_spaces,
        target_dim_range=target_range
    )
    
    # 模拟训练过程中的维度变化
    x = torch.randn(batch_size, input_dim)
    
    print("训练过程中的维度变化:")
    dims_history = []
    
    for epoch in [0, 10, 50, 100, 200]:
        logits, info = model(x, epoch=epoch)
        dims = info['current_dims']
        dims_history.append(dims)
        
        print(f"Epoch {epoch:3d}: 维度 = {dims}, 平均 = {np.mean(dims):.1f}")
    
    # 验证维度在合理范围内
    for dims in dims_history:
        for dim in dims:
            assert target_range[0] <= dim <= target_range[1]
    
    print("✓ 维度演化测试通过")


def test_curvature_learning():
    """测试曲率学习"""
    print("\n=== 测试曲率学习 ===")
    
    input_dim = 64
    target_range = (16, 32)
    
    # 测试不同流形类型的曲率学习
    for manifold_type in ['hyperbolic', 'spherical']:
        print(f"\n测试 {manifold_type} 流形:")
        
        proj = MultiSpaceManifoldProjection(
            input_dim=input_dim,
            space_id=0,
            manifold_type=manifold_type,
            init_curvature=1.0,
            target_dim_range=target_range
        )
        
        print(f"  初始曲率: {proj.curvature:.4f}")
        
        # 模拟梯度更新
        x = torch.randn(8, input_dim)
        feat, manifold, dim = proj.project_to_manifold(x)
        
        # 计算一个简单的损失（特征的范数）
        loss = feat.norm(dim=1).mean()
        loss.backward()
        
        # 检查曲率参数是否有梯度
        if proj.log_curvature is not None:
            print(f"  曲率梯度: {proj.log_curvature.grad}")
            assert proj.log_curvature.grad is not None
        
        print(f"  损失: {loss.item():.4f}")
    
    print("✓ 曲率学习测试通过")


def visualize_space_usage():
    """可视化空间使用情况"""
    print("\n=== 可视化空间使用情况 ===")
    
    batch_size = 32
    input_dim = 128
    num_classes = 10
    num_spaces = 10
    target_range = (32, 128)
    
    base_model = DummyBaseModel(input_dim)
    model = create_multi_space_manifold_model(
        base_model=base_model,
        input_dim=input_dim,
        num_classes=num_classes,
        num_spaces=num_spaces,
        target_dim_range=target_range
    )
    
    # 生成测试数据
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        logits, info = model(x, epoch=0)
    
    # 提取信息
    dims = info['current_dims']
    curvatures = info['curvatures']
    gates = info['fusion_gates'].mean(dim=0).cpu().numpy()
    
    print("空间使用统计:")
    print(f"  维度分布: min={min(dims)}, max={max(dims)}, avg={np.mean(dims):.1f}")
    print(f"  曲率分布: min={min([c.item() if isinstance(c, torch.Tensor) else c for c in curvatures]):.3f}, "
          f"max={max([c.item() if isinstance(c, torch.Tensor) else c for c in curvatures]):.3f}")
    print(f"  门控权重: {[f'{g:.3f}' for g in gates]}")
    
    # 创建可视化（如果在支持的环境中）
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # 维度分布
        ax1.bar(range(num_spaces), dims)
        ax1.set_title('各空间的维度')
        ax1.set_xlabel('空间ID')
        ax1.set_ylabel('维度')
        
        # 曲率分布
        curv_values = [c.item() if isinstance(c, torch.Tensor) else c for c in curvatures]
        ax2.bar(range(num_spaces), curv_values)
        ax2.set_title('各空间的曲率')
        ax2.set_xlabel('空间ID')
        ax2.set_ylabel('曲率')
        
        # 门控权重分布
        ax3.bar(range(num_spaces), gates)
        ax3.set_title('门控权重分布')
        ax3.set_xlabel('空间ID')
        ax3.set_ylabel('权重')
        
        plt.tight_layout()
        plt.savefig('multi_space_analysis.png', dpi=150, bbox_inches='tight')
        print("  可视化保存到: multi_space_analysis.png")
        
    except Exception as e:
        print(f"  可视化跳过 (原因: {e})")
    
    print("✓ 空间使用分析完成")


def run_all_tests():
    """运行所有测试"""
    print("开始多空间可学习流形模型测试\n")
    
    try:
        test_adaptive_dimension_projection()
        test_multi_space_manifold_projection()
        test_multi_space_network()
        test_loss_function()
        test_dimension_evolution()
        test_curvature_learning()
        visualize_space_usage()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！多空间架构工作正常。")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()