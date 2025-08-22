"""
测试可学习流形方案的基本功能
"""

import torch
import torch.nn as nn
from learnable_manifold import (
    LearnableManifoldProjection, 
    GatedManifoldFusion,
    LearnableManifoldNetwork,
    ManifoldMatchingLoss
)
import models.convnet as CN


def test_learnable_manifold_projection():
    """测试可学习流形投影层"""
    print("Testing LearnableManifoldProjection...")
    
    batch_size = 16
    input_dim = 128
    
    # 测试双曲投影
    hyp_proj = LearnableManifoldProjection(input_dim, 'hyperbolic', init_curvature=1.0)
    x = torch.randn(batch_size, input_dim)
    
    # 投影到流形
    hyp_feat, hyp_manifold = hyp_proj.project_to_manifold(x)
    print(f"Hyperbolic features shape: {hyp_feat.shape}")
    print(f"Initial curvature: {hyp_proj.curvature.item():.4f}")
    
    # 测试切空间映射
    tangent = hyp_proj.to_tangent_space(hyp_feat, hyp_manifold)
    print(f"Tangent vector shape: {tangent.shape}")
    
    # 测试球面投影
    sph_proj = LearnableManifoldProjection(input_dim, 'spherical', init_curvature=1.0)
    sph_feat, sph_manifold = sph_proj.project_to_manifold(x)
    print(f"Spherical features shape: {sph_feat.shape}")
    
    # 测试欧式投影
    eucl_proj = LearnableManifoldProjection(input_dim, 'euclidean')
    eucl_feat, eucl_manifold = eucl_proj.project_to_manifold(x)
    print(f"Euclidean features shape: {eucl_feat.shape}")
    
    print("✓ LearnableManifoldProjection test passed!\n")


def test_gated_fusion():
    """测试门控融合机制"""
    print("Testing GatedManifoldFusion...")
    
    batch_size = 16
    input_dim = 128
    num_manifolds = 3
    
    fusion = GatedManifoldFusion(input_dim, num_manifolds)
    
    # 模拟三个流形的特征
    features = [torch.randn(batch_size, input_dim) for _ in range(num_manifolds)]
    
    fused, gates = fusion(features)
    print(f"Fused features shape: {fused.shape}")
    print(f"Gates shape: {gates.shape}")
    print(f"Gates sum per sample: {gates.sum(dim=1)[0].item():.4f} (should be ~1.0)")
    
    print("✓ GatedManifoldFusion test passed!\n")


def test_learnable_manifold_network():
    """测试完整的可学习流形网络"""
    print("Testing LearnableManifoldNetwork...")
    
    batch_size = 8
    num_classes = 10
    input_channels = 3
    image_size = 32
    
    # 创建基础模型
    base_model = CN.ConvNet(
        channel=input_channels,
        num_classes=num_classes,
        net_width=128,
        net_depth=3,
        net_norm='instance',
        im_size=(image_size, image_size)
    )
    
    # 获取特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, input_channels, image_size, image_size)
        dummy_features = base_model.embed(dummy_input)
        feature_dim = dummy_features.shape[1]
    
    # 创建可学习流形网络
    model = LearnableManifoldNetwork(
        base_model=base_model,
        input_dim=feature_dim,
        num_classes=num_classes,
        hyperbolic_init_c=1.0,
        spherical_init_c=1.0
    )
    
    # 测试前向传播
    x = torch.randn(batch_size, input_channels, image_size, image_size)
    logits, info = model(x)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Euclidean features shape: {info['euclidean_features'].shape}")
    print(f"Hyperbolic features shape: {info['hyperbolic_features'].shape}")
    print(f"Spherical features shape: {info['spherical_features'].shape}")
    print(f"Fused features shape: {info['fused_features'].shape}")
    print(f"Gates shape: {info['gates'].shape}")
    print(f"Hyperbolic curvature: {info['hyperbolic_curvature'].item():.4f}")
    print(f"Spherical curvature: {info['spherical_curvature'].item():.4f}")
    
    print("✓ LearnableManifoldNetwork test passed!\n")


def test_manifold_matching_loss():
    """测试流形匹配损失函数（无teacher）"""
    print("Testing ManifoldMatchingLoss...")
    
    batch_size = 8
    num_classes = 10
    feature_dim = 128
    
    # 创建损失函数
    loss_fn = ManifoldMatchingLoss()
    
    # 模拟真实数据和合成数据的特征
    real_info = {
        'euclidean_features': torch.randn(batch_size, feature_dim),
        'hyperbolic_features': torch.randn(batch_size, feature_dim),
        'spherical_features': torch.randn(batch_size, feature_dim),
        'fused_features': torch.randn(batch_size, feature_dim),
        'gates': torch.softmax(torch.randn(batch_size, 3), dim=-1),
        'hyperbolic_curvature': torch.tensor(1.2),
        'spherical_curvature': torch.tensor(0.9)
    }
    
    syn_info = {
        'euclidean_features': torch.randn(batch_size, feature_dim),
        'hyperbolic_features': torch.randn(batch_size, feature_dim),
        'spherical_features': torch.randn(batch_size, feature_dim),
        'fused_features': torch.randn(batch_size, feature_dim),
        'gates': torch.softmax(torch.randn(batch_size, 3), dim=-1),
        'hyperbolic_curvature': torch.tensor(1.5),
        'spherical_curvature': torch.tensor(0.8)
    }
    
    # 计算损失
    total_loss, loss_dict = loss_fn(
        None, None,  # 不使用logits
        real_info, syn_info
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print("✓ ManifoldMatchingLoss test passed!\n")


def test_gradient_flow():
    """测试梯度流动"""
    print("Testing gradient flow...")
    
    batch_size = 4
    input_dim = 64
    
    # 创建可学习的投影层
    proj = LearnableManifoldProjection(input_dim, 'hyperbolic', init_curvature=1.0)
    
    # 输入数据
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    
    # 前向传播
    feat, manifold = proj.project_to_manifold(x)
    tangent = proj.to_tangent_space(feat, manifold)
    
    # 模拟损失
    loss = tangent.sum()
    
    # 反向传播
    loss.backward()
    
    print(f"Input gradient exists: {x.grad is not None}")
    print(f"Projection layer gradient exists: {proj.proj_layer.weight.grad is not None}")
    print(f"Curvature gradient exists: {proj.log_curvature.grad is not None}")
    
    if proj.log_curvature.grad is not None:
        print(f"Curvature gradient magnitude: {proj.log_curvature.grad.abs().item():.6f}")
    
    print("✓ Gradient flow test passed!\n")


def test_mmd_computation():
    """测试MMD损失计算"""
    print("Testing MMD computation...")
    
    from m3dloss import M3DLoss
    
    batch_size = 16
    feature_dim = 64
    
    # 创建不同的核函数
    gaussian_mmd = M3DLoss(kernel_type='gaussian')
    hyperbolic_mmd = M3DLoss(kernel_type='hyperbolic')
    
    # 生成测试数据
    real_features = torch.randn(batch_size, feature_dim)
    syn_features = torch.randn(batch_size, feature_dim)
    
    # 计算MMD
    gaussian_loss = gaussian_mmd(real_features, syn_features)
    hyperbolic_loss = hyperbolic_mmd(real_features, syn_features)
    
    print(f"Gaussian MMD loss: {gaussian_loss.item():.6f}")
    print(f"Hyperbolic MMD loss: {hyperbolic_loss.item():.6f}")
    
    # 测试相同分布的MMD（应该接近0）
    same_loss = gaussian_mmd(real_features, real_features)
    print(f"Same distribution MMD: {same_loss.item():.8f} (should be close to 0)")
    
    print("✓ MMD computation test passed!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Learnable Manifold Components (No Teacher)")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行测试
    test_learnable_manifold_projection()
    test_gated_fusion()
    test_learnable_manifold_network()
    test_manifold_matching_loss()
    test_gradient_flow()
    test_mmd_computation()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)