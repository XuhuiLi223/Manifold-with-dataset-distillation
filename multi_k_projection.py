import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds import PoincareBall, Sphere, Euclidean, ProductManifold
from geoopt.optim import RiemannianAdam
import numpy as np
from typing import List, Tuple, Dict, Optional


class DimensionReducer(nn.Module):
    """维度降维器，支持从高维逐步降到目标维度"""
    
    def __init__(self, input_dim: int, target_dim: int, num_stages: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.num_stages = num_stages
        
        # 计算每个阶段的维度
        dims = np.linspace(input_dim, target_dim, num_stages + 1).astype(int)
        
        # 构建降维网络
        layers = []
        for i in range(num_stages):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU() if i < num_stages - 1 else nn.Identity()
            ])
        
        self.reducer = nn.Sequential(*layers)
        
        # 可学习的维度重要性权重
        self.dimension_weights = nn.Parameter(torch.ones(target_dim))
        
    def forward(self, x):
        """前向传播，返回降维后的特征和维度权重"""
        reduced = self.reducer(x)
        # 应用维度权重
        weighted = reduced * torch.softmax(self.dimension_weights, dim=0)
        return weighted, self.dimension_weights


class MultiSpaceProjection(nn.Module):
    """多空间投影层，支持多种流形类型"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 manifold_type: str = 'mixed', 
                 init_curvature: float = 1.0,
                 learnable_dim: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.manifold_type = manifold_type
        self.learnable_dim = learnable_dim
        
        # 可学习的曲率参数（对数尺度）
        self.log_curvature = nn.Parameter(torch.tensor(np.log(init_curvature)))
        
        # 维度降维器
        if learnable_dim:
            self.dim_reducer = DimensionReducer(input_dim, output_dim)
        else:
            self.proj_layer = nn.Linear(input_dim, output_dim)
        
        # 流形特定的投影参数
        if manifold_type == 'hyperbolic':
            # 双曲空间特定参数
            self.hyperbolic_bias = nn.Parameter(torch.zeros(output_dim))
        elif manifold_type == 'spherical':
            # 球面空间特定参数
            self.sphere_scale = nn.Parameter(torch.ones(1))
        elif manifold_type == 'mixed':
            # 混合流形参数
            self.mix_weight = nn.Parameter(torch.tensor(0.5))
        
    @property
    def curvature(self):
        """获取当前曲率（保证为正）"""
        return torch.exp(self.log_curvature).clamp(min=1e-6, max=10.0)
    
    def project_to_manifold(self, x):
        """将欧式空间的点投影到流形上"""
        # 维度变换
        if self.learnable_dim:
            x, dim_weights = self.dim_reducer(x)
        else:
            x = self.proj_layer(x)
            dim_weights = None
        
        if self.manifold_type == 'hyperbolic':
            # 投影到Poincaré ball
            manifold = PoincareBall(c=self.curvature)
            x = x + self.hyperbolic_bias
            # 确保点在Poincaré ball内部
            norm = x.norm(dim=-1, keepdim=True)
            x = x / (norm + 1e-5) * torch.tanh(norm / 2)
            x = manifold.projx(x)
            
        elif self.manifold_type == 'spherical':
            # 投影到球面
            manifold = Sphere()
            x = F.normalize(x, p=2, dim=-1) * self.sphere_scale * torch.sqrt(1.0 / self.curvature)
            x = manifold.projx(x)
            
        elif self.manifold_type == 'euclidean':
            # 欧式空间
            manifold = Euclidean()
            
        elif self.manifold_type == 'mixed':
            # 混合流形（双曲和球面的组合）
            weight = torch.sigmoid(self.mix_weight)
            
            # 双曲部分
            hyp_manifold = PoincareBall(c=self.curvature)
            norm = x.norm(dim=-1, keepdim=True)
            hyp_x = x / (norm + 1e-5) * torch.tanh(norm / 2)
            hyp_x = hyp_manifold.projx(hyp_x)
            
            # 球面部分
            sph_manifold = Sphere()
            sph_x = F.normalize(x, p=2, dim=-1) * torch.sqrt(1.0 / self.curvature)
            sph_x = sph_manifold.projx(sph_x)
            
            # 加权组合
            x = weight * hyp_x + (1 - weight) * sph_x
            manifold = ProductManifold((hyp_manifold, sph_manifold))
            
        else:
            raise ValueError(f"Unknown manifold type: {self.manifold_type}")
            
        return x, manifold, dim_weights


class MultiKProjectionNetwork(nn.Module):
    """多K投影网络，支持10个可学习的流形空间"""
    
    def __init__(self, base_model, input_dim: int, num_classes: int,
                 num_spaces: int = 10,
                 initial_dims: List[int] = None,
                 manifold_types: List[str] = None,
                 init_curvatures: List[float] = None):
        super().__init__()
        
        self.base_model = base_model
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_spaces = num_spaces
        
        # 设置默认值
        if initial_dims is None:
            # 从高维开始，逐步降低
            initial_dims = [input_dim - i * (input_dim // (num_spaces + 1)) for i in range(num_spaces)]
        
        if manifold_types is None:
            # 混合不同类型的流形
            types = ['hyperbolic', 'spherical', 'euclidean', 'mixed']
            manifold_types = [types[i % len(types)] for i in range(num_spaces)]
        
        if init_curvatures is None:
            # 不同的初始曲率
            init_curvatures = [0.1 + 0.2 * i for i in range(num_spaces)]
        
        # 创建多个投影空间
        self.projections = nn.ModuleList([
            MultiSpaceProjection(
                input_dim=input_dim,
                output_dim=initial_dims[i],
                manifold_type=manifold_types[i],
                init_curvature=init_curvatures[i],
                learnable_dim=True
            )
            for i in range(num_spaces)
        ])
        
        # 空间重要性学习网络
        self.space_importance = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_spaces),
            nn.Softmax(dim=-1)
        )
        
        # 多空间融合器
        self.fusion_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(initial_dims[i], 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            for i in range(num_spaces)
        ])
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(64 * num_spaces, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Linear(128, num_classes)
        
        # 存储各空间的统计信息
        self.register_buffer('space_utilization', torch.zeros(num_spaces))
        
    def embed(self, x):
        """提取基础特征"""
        return self.base_model.embed(x)
    
    def forward(self, x, return_all_features=True):
        """前向传播"""
        # 1. 提取基础特征
        base_features = self.embed(x)  # [B, D]
        
        # 2. 计算空间重要性权重
        space_weights = self.space_importance(base_features)  # [B, num_spaces]
        
        # 3. 投影到多个流形空间
        projected_features = []
        manifolds = []
        dim_weights_list = []
        curvatures = []
        
        for i, proj in enumerate(self.projections):
            proj_feat, manifold, dim_weights = proj.project_to_manifold(base_features)
            projected_features.append(proj_feat)
            manifolds.append(manifold)
            dim_weights_list.append(dim_weights)
            curvatures.append(proj.curvature)
        
        # 4. 特征融合
        fused_features = []
        for i, (feat, fuser) in enumerate(zip(projected_features, self.fusion_network)):
            # 应用空间权重
            weighted_feat = feat * space_weights[:, i:i+1]
            # 通过融合网络
            fused = fuser(weighted_feat)
            fused_features.append(fused)
        
        # 5. 拼接所有空间的特征
        concat_features = torch.cat(fused_features, dim=1)  # [B, 64 * num_spaces]
        
        # 6. 最终融合
        final_features = self.final_fusion(concat_features)  # [B, 128]
        
        # 7. 分类
        logits = self.classifier(final_features)
        
        # 更新空间利用率统计
        with torch.no_grad():
            self.space_utilization = 0.9 * self.space_utilization + 0.1 * space_weights.mean(dim=0)
        
        if return_all_features:
            return logits, {
                'base_features': base_features,
                'projected_features': projected_features,
                'space_weights': space_weights,
                'final_features': final_features,
                'curvatures': torch.stack(curvatures),
                'dim_weights': dim_weights_list,
                'space_utilization': self.space_utilization,
                'manifolds': manifolds
            }
        else:
            return logits


class MultiKProjectionLoss(nn.Module):
    """多K投影损失函数"""
    
    def __init__(self, num_spaces: int = 10):
        super().__init__()
        self.num_spaces = num_spaces
        
    def compute_multi_space_mmd(self, real_features: List[torch.Tensor], 
                                syn_features: List[torch.Tensor],
                                curvatures: torch.Tensor):
        """计算多空间MMD损失"""
        from m3dloss import M3DLoss
        
        total_loss = 0.0
        loss_dict = {}
        
        for i, (real_feat, syn_feat) in enumerate(zip(real_features, syn_features)):
            # 根据曲率选择合适的核函数
            if curvatures[i] > 1.0:  # 高曲率，使用双曲核
                m3d = M3DLoss(kernel_type='hyperbolic')
            else:  # 低曲率，使用高斯核
                m3d = M3DLoss(kernel_type='gaussian')
            
            space_loss = m3d(real_feat, syn_feat)
            total_loss += space_loss
            loss_dict[f'space_{i}_mmd'] = space_loss.item()
        
        return total_loss / self.num_spaces, loss_dict
    
    def compute_space_diversity_loss(self, space_weights: torch.Tensor):
        """计算空间多样性损失，鼓励使用所有空间"""
        # 计算熵
        entropy = -torch.sum(space_weights * torch.log(space_weights + 1e-8), dim=1).mean()
        # 最大化熵（负熵作为损失）
        return -entropy
    
    def compute_dimension_sparsity_loss(self, dim_weights_list: List[torch.Tensor]):
        """计算维度稀疏性损失，鼓励维度选择的稀疏性"""
        total_sparsity = 0.0
        
        for dim_weights in dim_weights_list:
            if dim_weights is not None:
                # L1正则化促进稀疏性
                sparsity = torch.abs(dim_weights).mean()
                total_sparsity += sparsity
        
        return total_sparsity / len(dim_weights_list)
    
    def compute_curvature_regularization(self, curvatures: torch.Tensor):
        """曲率正则化"""
        # 鼓励曲率在合理范围内 [0.01, 10.0]
        reg = torch.relu(0.01 - curvatures).mean() + torch.relu(curvatures - 10.0).mean()
        
        # 鼓励曲率的多样性
        curvature_std = torch.std(curvatures)
        diversity_reg = torch.relu(0.5 - curvature_std)  # 标准差至少为0.5
        
        return reg + 0.1 * diversity_reg
    
    def forward(self, real_output: Dict, syn_output: Dict):
        """计算总损失"""
        total_loss = 0.0
        loss_dict = {}
        
        # 1. 多空间MMD损失
        mmd_loss, mmd_dict = self.compute_multi_space_mmd(
            real_output['projected_features'],
            syn_output['projected_features'],
            syn_output['curvatures']
        )
        total_loss += mmd_loss
        loss_dict.update(mmd_dict)
        
        # 2. 空间多样性损失
        diversity_loss = self.compute_space_diversity_loss(syn_output['space_weights'])
        total_loss += 0.1 * diversity_loss
        loss_dict['space_diversity'] = diversity_loss.item()
        
        # 3. 维度稀疏性损失
        if syn_output['dim_weights'][0] is not None:
            sparsity_loss = self.compute_dimension_sparsity_loss(syn_output['dim_weights'])
            total_loss += 0.01 * sparsity_loss
            loss_dict['dim_sparsity'] = sparsity_loss.item()
        
        # 4. 曲率正则化
        curv_reg = self.compute_curvature_regularization(syn_output['curvatures'])
        total_loss += 0.01 * curv_reg
        loss_dict['curvature_reg'] = curv_reg.item()
        
        # 5. 最终特征MMD
        from m3dloss import M3DLoss
        final_m3d = M3DLoss(kernel_type='gaussian')
        final_loss = final_m3d(real_output['final_features'], syn_output['final_features'])
        total_loss += 2.0 * final_loss  # 更高权重
        loss_dict['final_mmd'] = final_loss.item()
        
        return total_loss, loss_dict


def create_multi_k_projection_model(base_model, input_dim: int, num_classes: int,
                                   num_spaces: int = 10, **kwargs):
    """创建多K投影模型的便捷函数"""
    return MultiKProjectionNetwork(
        base_model=base_model,
        input_dim=input_dim,
        num_classes=num_classes,
        num_spaces=num_spaces,
        **kwargs
    )