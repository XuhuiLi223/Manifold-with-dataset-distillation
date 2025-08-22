import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds import PoincareBall, Sphere, Euclidean
from geoopt.optim import RiemannianAdam
import numpy as np


class LearnableManifoldProjection(nn.Module):
    """可学习的流形投影层，支持动态调整曲率"""
    
    def __init__(self, input_dim, manifold_type='hyperbolic', init_curvature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.manifold_type = manifold_type
        
        # 可学习的曲率参数
        self.log_curvature = nn.Parameter(torch.tensor(np.log(init_curvature)))
        
        # 投影到流形的线性变换
        self.proj_layer = nn.Linear(input_dim, input_dim)
        
    @property
    def curvature(self):
        """获取当前曲率（保证为正）"""
        return torch.exp(self.log_curvature).clamp(min=1e-6, max=10.0)
    
    def project_to_manifold(self, x):
        """将欧式空间的点投影到流形上"""
        # 先进行线性变换
        x = self.proj_layer(x)
        
        if self.manifold_type == 'hyperbolic':
            # 投影到Poincaré ball
            manifold = PoincareBall(c=self.curvature)
            # 确保点在Poincaré ball内部
            norm = x.norm(dim=-1, keepdim=True)
            x = x / (norm + 1e-5) * torch.tanh(norm / 2)
            x = manifold.projx(x)
        elif self.manifold_type == 'spherical':
            # 投影到球面
            manifold = Sphere()
            x = F.normalize(x, p=2, dim=-1) * torch.sqrt(1.0 / self.curvature)
            x = manifold.projx(x)
        else:  # euclidean
            manifold = Euclidean()
            x = x
            
        return x, manifold
    
    def to_tangent_space(self, x, manifold, base_point=None):
        """将流形上的点映射到切空间"""
        if base_point is None:
            # 默认使用原点作为切点
            base_point = manifold.origin(x.shape[0], self.input_dim).to(x.device)
        
        # 使用对数映射将点映射到切空间
        tangent_vec = manifold.logmap(base_point, x)
        return tangent_vec
    
    def from_tangent_space(self, tangent_vec, manifold, base_point=None):
        """从切空间映射回流形"""
        if base_point is None:
            base_point = manifold.origin(tangent_vec.shape[0], self.input_dim).to(tangent_vec.device)
        
        # 使用指数映射将切向量映射回流形
        x = manifold.expmap(base_point, tangent_vec)
        return x


class GatedManifoldFusion(nn.Module):
    """门控机制融合多个流形空间的特征"""
    
    def __init__(self, input_dim, num_manifolds=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_manifolds = num_manifolds
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * num_manifolds, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_manifolds),
            nn.Softmax(dim=-1)
        )
        
        # 特征融合层
        self.fusion_layer = nn.Linear(input_dim, input_dim)
        
    def forward(self, features_list):
        """
        features_list: list of tensors, each of shape [B, D]
        返回融合后的特征 [B, D]
        """
        # 拼接所有特征
        concat_features = torch.cat(features_list, dim=-1)  # [B, D*num_manifolds]
        
        # 计算门控权重
        gates = self.gate_net(concat_features)  # [B, num_manifolds]
        
        # 加权融合
        fused = torch.zeros_like(features_list[0])
        for i, feat in enumerate(features_list):
            fused += gates[:, i:i+1] * feat
        
        # 最终变换
        fused = self.fusion_layer(fused)
        return fused, gates


class LearnableManifoldNetwork(nn.Module):
    """可学习的多流形网络"""
    
    def __init__(self, base_model, input_dim, num_classes, 
                 hyperbolic_init_c=1.0, spherical_init_c=1.0):
        super().__init__()
        
        # 基础特征提取器
        self.base_model = base_model
        self.input_dim = input_dim
        
        # 三个流形投影：欧式（固定）、双曲（可学）、球面（可学）
        self.euclidean_proj = LearnableManifoldProjection(
            input_dim, manifold_type='euclidean'
        )
        self.hyperbolic_proj = LearnableManifoldProjection(
            input_dim, manifold_type='hyperbolic', init_curvature=hyperbolic_init_c
        )
        self.spherical_proj = LearnableManifoldProjection(
            input_dim, manifold_type='spherical', init_curvature=spherical_init_c
        )
        
        # 门控融合机制
        self.gate_fusion = GatedManifoldFusion(input_dim, num_manifolds=3)
        
        # 最终分类器（在双曲空间）
        self.final_manifold = PoincareBall(c=1.0)
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def embed(self, x):
        """提取基础特征"""
        return self.base_model.embed(x)
    
    def forward(self, x):
        # 1. 提取基础特征（欧式空间）
        base_features = self.embed(x)  # [B, D]
        
        # 2. 投影到三个流形空间
        eucl_feat, eucl_manifold = self.euclidean_proj.project_to_manifold(base_features)
        hyp_feat, hyp_manifold = self.hyperbolic_proj.project_to_manifold(base_features)
        sph_feat, sph_manifold = self.spherical_proj.project_to_manifold(base_features)
        
        # 3. 映射到共同的切空间（以欧式空间原点为切点）
        origin = torch.zeros_like(base_features)
        eucl_tangent = eucl_feat  # 欧式空间的切空间就是自身
        hyp_tangent = self.hyperbolic_proj.to_tangent_space(hyp_feat, hyp_manifold, origin)
        sph_tangent = self.spherical_proj.to_tangent_space(sph_feat, sph_manifold, origin)
        
        # 4. 在切空间进行门控融合
        tangent_features = [eucl_tangent, hyp_tangent, sph_tangent]
        fused_tangent, gates = self.gate_fusion(tangent_features)
        
        # 5. 映射回目标流形（双曲空间）
        final_features = self.final_manifold.expmap(origin, fused_tangent)
        final_features = self.final_manifold.projx(final_features)
        
        # 6. 分类
        # 在双曲空间中进行分类需要特殊处理
        # 使用Möbius线性层或先映射回欧式空间
        logits = self.classifier(self.final_manifold.logmap0(final_features))
        
        return logits, {
            'euclidean_features': eucl_feat,
            'hyperbolic_features': hyp_feat,
            'spherical_features': sph_feat,
            'fused_features': final_features,
            'gates': gates,
            'hyperbolic_curvature': self.hyperbolic_proj.curvature,
            'spherical_curvature': self.spherical_proj.curvature
        }


class ManifoldMatchingLoss(nn.Module):
    """多流形匹配损失（不需要teacher model）"""
    
    def __init__(self):
        super().__init__()
        
    def compute_manifold_alignment_loss(self, features_dict, real_features, syn_features):
        """计算流形对齐损失"""
        total_loss = 0.0
        loss_dict = {}
        
        # 1. 真实数据和合成数据在各个流形空间的MMD损失
        from m3dloss import M3DLoss
        
        # 欧式空间MMD
        eucl_m3d = M3DLoss(kernel_type='gaussian')
        eucl_loss = eucl_m3d(real_features['euclidean_features'], syn_features['euclidean_features'])
        total_loss += eucl_loss
        loss_dict['eucl_mmd'] = eucl_loss.item()
        
        # 双曲空间MMD
        hyp_m3d = M3DLoss(kernel_type='hyperbolic')
        hyp_loss = hyp_m3d(real_features['hyperbolic_features'], syn_features['hyperbolic_features'])
        total_loss += hyp_loss
        loss_dict['hyp_mmd'] = hyp_loss.item()
        
        # 球面空间MMD（使用高斯核近似）
        sph_m3d = M3DLoss(kernel_type='gaussian')
        sph_loss = sph_m3d(real_features['spherical_features'], syn_features['spherical_features'])
        total_loss += sph_loss
        loss_dict['sph_mmd'] = sph_loss.item()
        
        # 2. 融合特征的MMD损失
        fused_m3d = M3DLoss(kernel_type='gaussian')
        fused_loss = fused_m3d(real_features['fused_features'], syn_features['fused_features'])
        total_loss += fused_loss * 2.0  # 融合特征更重要，权重更高
        loss_dict['fused_mmd'] = fused_loss.item()
        
        return total_loss, loss_dict
    
    def compute_gate_consistency_loss(self, real_gates, syn_gates):
        """计算门控一致性损失（鼓励相似的数据有相似的门控权重）"""
        # 使用L2损失
        gate_loss = F.mse_loss(syn_gates, real_gates.detach())
        return gate_loss
    
    def compute_curvature_regularization(self, hyp_curvature, sph_curvature):
        """曲率正则化，防止曲率过大或过小"""
        # 鼓励曲率在合理范围内 [0.1, 5.0]
        hyp_reg = torch.relu(0.1 - hyp_curvature) + torch.relu(hyp_curvature - 5.0)
        sph_reg = torch.relu(0.1 - sph_curvature) + torch.relu(sph_curvature - 5.0)
        return hyp_reg + sph_reg
    
    def forward(self, real_output, syn_output, real_info, syn_info):
        """
        计算总的损失（不使用teacher model）
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 1. 分类损失（如果有标签）
        if real_output is not None and syn_output is not None:
            # 可以使用交叉熵或其他分类损失
            pass
        
        # 2. 流形对齐损失（主要损失）
        alignment_loss, alignment_dict = self.compute_manifold_alignment_loss(
            {}, real_info, syn_info
        )
        total_loss += alignment_loss
        loss_dict.update(alignment_dict)
        
        # 3. 门控一致性损失
        gate_loss = self.compute_gate_consistency_loss(
            real_info['gates'], syn_info['gates']
        )
        total_loss += gate_loss * 0.1
        loss_dict['gate_loss'] = gate_loss.item()
        
        # 4. 曲率正则化
        curv_reg = self.compute_curvature_regularization(
            syn_info['hyperbolic_curvature'],
            syn_info['spherical_curvature']
        )
        total_loss += curv_reg * 0.01
        loss_dict['curvature_reg'] = curv_reg.item()
        
        # 5. 门控熵正则化（鼓励探索不同的流形）
        gate_entropy = -torch.sum(syn_info['gates'] * torch.log(syn_info['gates'] + 1e-8), dim=1).mean()
        entropy_reg = -gate_entropy  # 最大化熵
        total_loss += entropy_reg * 0.01
        loss_dict['entropy_reg'] = entropy_reg.item()
        
        return total_loss, loss_dict


def create_learnable_manifold_model(base_model, input_dim, num_classes):
    """创建可学习流形模型的便捷函数"""
    return LearnableManifoldNetwork(
        base_model=base_model,
        input_dim=input_dim,
        num_classes=num_classes,
        hyperbolic_init_c=1.0,
        spherical_init_c=1.0
    )