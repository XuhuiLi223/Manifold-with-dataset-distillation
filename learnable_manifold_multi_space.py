import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds import PoincareBall, Sphere, Euclidean
from geoopt.optim import RiemannianAdam
import numpy as np
import math


class AdaptiveDimensionProjection(nn.Module):
    """自适应维度投影层，支持从高维到低维的可学习降维"""
    
    def __init__(self, input_dim, target_dim_range=(32, 256), init_target_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.min_dim = target_dim_range[0]
        self.max_dim = target_dim_range[1]
        
        # 初始目标维度
        if init_target_dim is None:
            init_target_dim = min(input_dim, (self.min_dim + self.max_dim) // 2)
        
        # 可学习的目标维度参数（使用sigmoid激活确保在合理范围内）
        self.dim_logit = nn.Parameter(torch.tensor(
            math.log(init_target_dim - self.min_dim + 1e-6) - 
            math.log(self.max_dim - init_target_dim + 1e-6)
        ))
        
        # 投影层（使用最大可能的维度）
        self.projection = nn.Linear(input_dim, self.max_dim)
        self.layer_norm = nn.LayerNorm(self.max_dim)
        
    @property
    def target_dim(self):
        """获取当前的目标维度"""
        # 使用sigmoid将logit转换为[min_dim, max_dim]范围内的整数
        prob = torch.sigmoid(self.dim_logit)
        dim_float = self.min_dim + prob * (self.max_dim - self.min_dim)
        return int(torch.round(dim_float).item())
    
    def forward(self, x):
        """前向传播，动态选择维度"""
        # 投影到最大维度
        projected = self.projection(x)  # [B, max_dim]
        projected = self.layer_norm(projected)
        
        # 动态截取到目标维度
        current_dim = self.target_dim
        if current_dim < self.max_dim:
            # 使用前current_dim个维度
            projected = projected[..., :current_dim]
        
        return projected, current_dim


class MultiSpaceManifoldProjection(nn.Module):
    """多空间流形投影层，支持10个不同的流形空间"""
    
    def __init__(self, input_dim, space_id, manifold_type='auto', 
                 init_curvature=1.0, target_dim_range=(32, 256)):
        super().__init__()
        self.input_dim = input_dim
        self.space_id = space_id
        self.target_dim_range = target_dim_range
        
        # 自适应维度投影
        self.dim_projection = AdaptiveDimensionProjection(
            input_dim, target_dim_range
        )
        
        # 根据space_id自动确定流形类型（如果未指定）
        if manifold_type == 'auto':
            manifold_types = [
                'euclidean', 'hyperbolic', 'spherical', 'euclidean',
                'hyperbolic', 'spherical', 'hyperbolic', 'spherical',
                'euclidean', 'hyperbolic'  # 10个空间的默认类型
            ]
            self.manifold_type = manifold_types[space_id % 10]
        else:
            self.manifold_type = manifold_type
        
        # 可学习的曲率参数（对于非欧式空间）
        if self.manifold_type != 'euclidean':
            self.log_curvature = nn.Parameter(torch.tensor(np.log(init_curvature)))
        else:
            self.register_parameter('log_curvature', None)
        
        # 空间特定的特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(target_dim_range[1], target_dim_range[1]),
            nn.ReLU(),
            nn.Linear(target_dim_range[1], target_dim_range[1])
        )
        
    @property
    def curvature(self):
        """获取当前曲率（保证为正）"""
        if self.log_curvature is None:
            return 1.0
        return torch.exp(self.log_curvature).clamp(min=1e-6, max=10.0)
    
    def project_to_manifold(self, x):
        """将欧式空间的点投影到流形上"""
        # 1. 自适应维度投影
        x, current_dim = self.dim_projection(x)  # [B, current_dim]
        
        # 2. 特征变换（padding到最大维度进行变换，然后截取）
        if current_dim < self.target_dim_range[1]:
            # 零填充到最大维度
            padded_x = F.pad(x, (0, self.target_dim_range[1] - current_dim))
            transformed = self.feature_transform(padded_x)
            x = transformed[..., :current_dim]  # 截取回当前维度
        else:
            x = self.feature_transform(x)
        
        # 3. 投影到指定流形
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
            # 欧式空间不需要特殊投影
            
        return x, manifold, current_dim
    
    def to_tangent_space(self, x, manifold, base_point=None):
        """将流形上的点映射到切空间"""
        if base_point is None:
            # 默认使用原点作为切点
            if hasattr(manifold, 'origin'):
                base_point = manifold.origin(x.shape[0], x.shape[-1]).to(x.device)
            else:
                base_point = torch.zeros_like(x)
        
        if self.manifold_type == 'euclidean':
            return x  # 欧式空间的切空间就是自身
        else:
            # 使用对数映射将点映射到切空间
            tangent_vec = manifold.logmap(base_point, x)
            return tangent_vec


class MultiSpaceGatedFusion(nn.Module):
    """多空间门控融合机制，支持10个流形空间的动态融合"""
    
    def __init__(self, max_dim, num_spaces=10, fusion_strategy='attention'):
        super().__init__()
        self.max_dim = max_dim
        self.num_spaces = num_spaces
        self.fusion_strategy = fusion_strategy
        
        if fusion_strategy == 'attention':
            # 注意力机制融合
            self.attention_net = nn.Sequential(
                nn.Linear(max_dim, max_dim // 2),
                nn.ReLU(),
                nn.Linear(max_dim // 2, 1)
            )
        elif fusion_strategy == 'gated':
            # 门控网络融合
            self.gate_net = nn.Sequential(
                nn.Linear(max_dim * num_spaces, max_dim),
                nn.ReLU(),
                nn.Linear(max_dim, num_spaces),
                nn.Softmax(dim=-1)
            )
        
        # 特征融合后的变换
        self.fusion_transform = nn.Sequential(
            nn.Linear(max_dim, max_dim),
            nn.LayerNorm(max_dim),
            nn.ReLU(),
            nn.Linear(max_dim, max_dim)
        )
        
    def forward(self, features_list, dims_list):
        """
        融合多个空间的特征
        features_list: list of tensors, 每个tensor形状为 [B, dim_i]
        dims_list: list of int, 每个空间的当前维度
        """
        batch_size = features_list[0].shape[0]
        
        # 将所有特征padding到相同的最大维度
        padded_features = []
        for feat, dim in zip(features_list, dims_list):
            if dim < self.max_dim:
                padded_feat = F.pad(feat, (0, self.max_dim - dim))
            else:
                padded_feat = feat
            padded_features.append(padded_feat)
        
        if self.fusion_strategy == 'attention':
            # 注意力机制融合
            attention_weights = []
            for feat in padded_features:
                weight = self.attention_net(feat)  # [B, 1]
                attention_weights.append(weight)
            
            attention_weights = torch.cat(attention_weights, dim=1)  # [B, num_spaces]
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # 加权融合
            fused = torch.zeros_like(padded_features[0])
            for i, feat in enumerate(padded_features):
                fused += attention_weights[:, i:i+1] * feat
            
            gates = attention_weights
            
        elif self.fusion_strategy == 'gated':
            # 门控网络融合
            concat_features = torch.cat(padded_features, dim=-1)  # [B, max_dim * num_spaces]
            gates = self.gate_net(concat_features)  # [B, num_spaces]
            
            # 加权融合
            fused = torch.zeros_like(padded_features[0])
            for i, feat in enumerate(padded_features):
                fused += gates[:, i:i+1] * feat
        
        # 最终变换
        fused = self.fusion_transform(fused)
        
        return fused, gates


class MultiSpaceLearnableManifoldNetwork(nn.Module):
    """多空间可学习流形网络，支持10个流形空间"""
    
    def __init__(self, base_model, input_dim, num_classes, 
                 num_spaces=10, target_dim_range=(32, 256),
                 init_curvatures=None, fusion_strategy='attention'):
        super().__init__()
        
        # 基础特征提取器
        self.base_model = base_model
        self.input_dim = input_dim
        self.num_spaces = num_spaces
        self.target_dim_range = target_dim_range
        
        # 初始化曲率
        if init_curvatures is None:
            init_curvatures = [0.5 + 0.5 * i for i in range(num_spaces)]  # 0.5到5.0
        
        # 创建多个流形投影空间
        self.manifold_projections = nn.ModuleList([
            MultiSpaceManifoldProjection(
                input_dim=input_dim,
                space_id=i,
                manifold_type='auto',
                init_curvature=init_curvatures[i % len(init_curvatures)],
                target_dim_range=target_dim_range
            ) for i in range(num_spaces)
        ])
        
        # 多空间门控融合
        self.multi_fusion = MultiSpaceGatedFusion(
            max_dim=target_dim_range[1],
            num_spaces=num_spaces,
            fusion_strategy=fusion_strategy
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(target_dim_range[1], target_dim_range[1] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim_range[1] // 2, num_classes)
        )
        
        # 维度自适应调节器
        self.dimension_scheduler = DimensionScheduler(
            num_spaces=num_spaces,
            target_dim_range=target_dim_range
        )
        
    def embed(self, x):
        """提取基础特征"""
        return self.base_model.embed(x)
    
    def forward(self, x, epoch=0):
        # 1. 提取基础特征
        base_features = self.embed(x)  # [B, input_dim]
        
        # 2. 投影到多个流形空间
        manifold_features = []
        manifold_info = []
        current_dims = []
        
        for i, proj in enumerate(self.manifold_projections):
            feat, manifold, dim = proj.project_to_manifold(base_features)
            
            # 映射到切空间
            tangent_feat = proj.to_tangent_space(feat, manifold)
            
            manifold_features.append(tangent_feat)
            manifold_info.append({
                'features': feat,
                'manifold': manifold,
                'curvature': proj.curvature if hasattr(proj, 'curvature') else 1.0,
                'manifold_type': proj.manifold_type
            })
            current_dims.append(dim)
        
        # 3. 多空间融合
        fused_features, fusion_gates = self.multi_fusion(manifold_features, current_dims)
        
        # 4. 分类
        logits = self.classifier(fused_features)
        
        # 5. 维度调度（可选的自适应调整）
        if self.training:
            self.dimension_scheduler.step(epoch, fusion_gates, manifold_info)
        
        return logits, {
            'manifold_features': [info['features'] for info in manifold_info],
            'manifold_info': manifold_info,
            'fused_features': fused_features,
            'fusion_gates': fusion_gates,
            'current_dims': current_dims,
            'curvatures': [info['curvature'] for info in manifold_info]
        }


class DimensionScheduler:
    """维度调度器，用于自适应调整各空间的维度"""
    
    def __init__(self, num_spaces, target_dim_range, 
                 reduction_rate=0.95, min_usage_threshold=0.1):
        self.num_spaces = num_spaces
        self.target_dim_range = target_dim_range
        self.reduction_rate = reduction_rate
        self.min_usage_threshold = min_usage_threshold
        
        # 跟踪各空间的使用情况
        self.usage_history = []
        self.step_count = 0
        
    def step(self, epoch, fusion_gates, manifold_info):
        """根据使用情况调整维度"""
        self.step_count += 1
        
        # 记录当前的门控权重（平均）
        avg_gates = fusion_gates.mean(dim=0).detach().cpu().numpy()
        self.usage_history.append(avg_gates)
        
        # 每100步进行一次调整
        if self.step_count % 100 == 0 and len(self.usage_history) >= 10:
            recent_usage = np.mean(self.usage_history[-10:], axis=0)
            
            for i in range(self.num_spaces):
                # 如果某个空间使用率很低，可以考虑降低其维度
                if recent_usage[i] < self.min_usage_threshold:
                    # 这里可以添加维度调整逻辑
                    pass


class MultiSpaceManifoldMatchingLoss(nn.Module):
    """多空间流形匹配损失"""
    
    def __init__(self, num_spaces=10):
        super().__init__()
        self.num_spaces = num_spaces
        
    def compute_multi_space_mmd_loss(self, real_info, syn_info):
        """计算多空间MMD损失"""
        from m3dloss import M3DLoss
        
        total_loss = 0.0
        loss_dict = {}
        
        # 对每个空间计算MMD损失
        for i in range(self.num_spaces):
            real_feat = real_info['manifold_features'][i]
            syn_feat = syn_info['manifold_features'][i]
            manifold_type = real_info['manifold_info'][i]['manifold_type']
            
            # 根据流形类型选择合适的核函数
            if manifold_type == 'hyperbolic':
                m3d = M3DLoss(kernel_type='hyperbolic')
            elif manifold_type == 'spherical':
                m3d = M3DLoss(kernel_type='gaussian')  # 球面用高斯核近似
            else:  # euclidean
                m3d = M3DLoss(kernel_type='gaussian')
            
            space_loss = m3d(real_feat, syn_feat)
            total_loss += space_loss
            loss_dict[f'space_{i}_mmd'] = space_loss.item()
        
        # 融合特征的MMD损失（权重更高）
        fused_m3d = M3DLoss(kernel_type='gaussian')
        fused_loss = fused_m3d(real_info['fused_features'], syn_info['fused_features'])
        total_loss += fused_loss * 2.0
        loss_dict['fused_mmd'] = fused_loss.item()
        
        return total_loss, loss_dict
    
    def compute_dimension_consistency_loss(self, real_info, syn_info):
        """维度一致性损失，鼓励相似的维度选择"""
        dim_loss = 0.0
        for i in range(self.num_spaces):
            real_dim = real_info['current_dims'][i]
            syn_dim = syn_info['current_dims'][i]
            # 简单的L1损失
            dim_loss += abs(real_dim - syn_dim) / max(real_dim, syn_dim)
        
        return dim_loss / self.num_spaces
    
    def compute_curvature_regularization(self, curvatures):
        """多空间曲率正则化"""
        reg_loss = 0.0
        for curvature in curvatures:
            if isinstance(curvature, torch.Tensor):
                # 鼓励曲率在合理范围内 [0.1, 5.0]
                reg_loss += torch.relu(0.1 - curvature) + torch.relu(curvature - 5.0)
        return reg_loss
    
    def compute_gate_diversity_loss(self, gates):
        """门控多样性损失，鼓励使用不同的空间"""
        # 计算熵，鼓励更均匀的分布
        entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=1).mean()
        return -entropy  # 最大化熵
    
    def forward(self, real_output, syn_output, real_info, syn_info):
        """计算总损失"""
        total_loss = 0.0
        loss_dict = {}
        
        # 1. 多空间MMD损失
        mmd_loss, mmd_dict = self.compute_multi_space_mmd_loss(real_info, syn_info)
        total_loss += mmd_loss
        loss_dict.update(mmd_dict)
        
        # 2. 维度一致性损失
        dim_loss = self.compute_dimension_consistency_loss(real_info, syn_info)
        total_loss += dim_loss * 0.1
        loss_dict['dim_consistency'] = dim_loss
        
        # 3. 曲率正则化
        curv_reg = self.compute_curvature_regularization(syn_info['curvatures'])
        total_loss += curv_reg * 0.01
        loss_dict['curvature_reg'] = curv_reg.item() if isinstance(curv_reg, torch.Tensor) else curv_reg
        
        # 4. 门控多样性损失
        gate_div = self.compute_gate_diversity_loss(syn_info['fusion_gates'])
        total_loss += gate_div * 0.01
        loss_dict['gate_diversity'] = gate_div.item()
        
        # 5. 门控一致性损失
        gate_consistency = F.mse_loss(syn_info['fusion_gates'], real_info['fusion_gates'].detach())
        total_loss += gate_consistency * 0.1
        loss_dict['gate_consistency'] = gate_consistency.item()
        
        return total_loss, loss_dict


def create_multi_space_manifold_model(base_model, input_dim, num_classes, 
                                      num_spaces=10, target_dim_range=(32, 256)):
    """创建多空间可学习流形模型的便捷函数"""
    return MultiSpaceLearnableManifoldNetwork(
        base_model=base_model,
        input_dim=input_dim,
        num_classes=num_classes,
        num_spaces=num_spaces,
        target_dim_range=target_dim_range,
        fusion_strategy='attention'
    )