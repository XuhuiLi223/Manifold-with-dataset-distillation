"""
多K投影方法演示脚本
此脚本展示了多K投影的核心概念，不依赖于PyTorch
"""

import numpy as np
from typing import List, Tuple, Dict


class MultiKProjectionDemo:
    """多K投影演示类"""
    
    def __init__(self, input_dim: int, num_spaces: int = 10):
        self.input_dim = input_dim
        self.num_spaces = num_spaces
        
        # 初始化每个空间的参数
        self.spaces = []
        for i in range(num_spaces):
            space = {
                'id': i,
                'type': ['hyperbolic', 'spherical', 'euclidean', 'mixed'][i % 4],
                'target_dim': input_dim - i * (input_dim // (num_spaces + 1)),
                'curvature': 0.1 + 0.2 * i,
                'importance': 1.0 / num_spaces  # 初始均匀重要性
            }
            self.spaces.append(space)
    
    def describe_architecture(self):
        """描述多K投影架构"""
        print("=" * 60)
        print("多K投影架构说明")
        print("=" * 60)
        print(f"\n输入维度: {self.input_dim}")
        print(f"投影空间数量: {self.num_spaces}")
        
        print("\n各投影空间配置:")
        print("-" * 60)
        print(f"{'空间ID':>6} | {'流形类型':>10} | {'目标维度':>8} | {'初始曲率':>8} | {'重要性':>8}")
        print("-" * 60)
        
        for space in self.spaces:
            print(f"{space['id']:>6} | {space['type']:>10} | "
                  f"{space['target_dim']:>8} | {space['curvature']:>8.2f} | "
                  f"{space['importance']:>8.3f}")
        
        print("-" * 60)
    
    def demonstrate_dimension_reduction(self):
        """演示维度降维过程"""
        print("\n" + "=" * 60)
        print("维度降维演示")
        print("=" * 60)
        
        # 模拟3阶段降维
        num_stages = 3
        
        for i, space in enumerate(self.spaces[:3]):  # 只演示前3个空间
            print(f"\n空间 {i} ({space['type']}):")
            dims = np.linspace(self.input_dim, space['target_dim'], num_stages + 1).astype(int)
            
            print(f"  降维路径: ", end="")
            for j in range(num_stages):
                print(f"{dims[j]} → ", end="")
            print(f"{dims[-1]}")
            
            # 模拟维度权重（稀疏性）
            dim_weights = np.random.exponential(0.5, space['target_dim'])
            dim_weights = dim_weights / dim_weights.sum()
            
            # 找出最重要的维度
            top_dims = np.argsort(dim_weights)[-5:]
            print(f"  最重要的5个维度: {top_dims}")
            print(f"  维度稀疏性: {np.std(dim_weights):.3f}")
    
    def demonstrate_multi_space_fusion(self):
        """演示多空间融合"""
        print("\n" + "=" * 60)
        print("多空间融合演示")
        print("=" * 60)
        
        # 模拟不同数据样本的空间权重
        num_samples = 5
        space_weights = np.random.dirichlet(np.ones(self.num_spaces), num_samples)
        
        print("\n不同样本的空间权重分布:")
        print("-" * 60)
        print("样本 | " + " | ".join([f"空间{i:2d}" for i in range(self.num_spaces)]))
        print("-" * 60)
        
        for i, weights in enumerate(space_weights):
            print(f"  {i}  | " + " | ".join([f"{w:6.3f}" for w in weights]))
        
        # 计算空间利用率
        avg_weights = space_weights.mean(axis=0)
        print("\n平均空间利用率:")
        for i, w in enumerate(avg_weights):
            bar = "█" * int(w * 50)
            print(f"  空间{i:2d}: {bar} {w:.3f}")
    
    def demonstrate_curvature_optimization(self):
        """演示曲率优化"""
        print("\n" + "=" * 60)
        print("曲率优化演示")
        print("=" * 60)
        
        # 模拟曲率优化过程
        iterations = 5
        print("\n曲率变化过程:")
        print("-" * 60)
        print("迭代 | " + " | ".join([f"空间{i}" for i in range(min(5, self.num_spaces))]))
        print("-" * 60)
        
        curvatures = np.array([s['curvature'] for s in self.spaces[:5]])
        
        for it in range(iterations):
            # 模拟曲率更新
            noise = np.random.normal(0, 0.1, len(curvatures))
            curvatures = np.clip(curvatures + noise, 0.01, 10.0)
            
            print(f" {it:3d} | " + " | ".join([f"{c:5.2f}" for c in curvatures]))
        
        print("\n曲率多样性指标:")
        print(f"  标准差: {np.std(curvatures):.3f}")
        print(f"  最小值: {np.min(curvatures):.3f}")
        print(f"  最大值: {np.max(curvatures):.3f}")
    
    def demonstrate_loss_components(self):
        """演示损失函数组件"""
        print("\n" + "=" * 60)
        print("损失函数组件")
        print("=" * 60)
        
        # 模拟各种损失值
        losses = {
            'space_0_mmd': 0.234,
            'space_1_mmd': 0.189,
            'space_2_mmd': 0.267,
            'space_3_mmd': 0.156,
            'space_4_mmd': 0.298,
            'final_mmd': 0.412,
            'space_diversity': 0.087,
            'dim_sparsity': 0.034,
            'curvature_reg': 0.021
        }
        
        print("\n损失分解:")
        total_loss = 0
        for name, value in losses.items():
            print(f"  {name:20s}: {value:.4f}")
            total_loss += value
        
        print("-" * 30)
        print(f"  {'总损失':20s}: {total_loss:.4f}")
        
        # 显示损失占比
        print("\n损失占比分析:")
        mmd_loss = sum(v for k, v in losses.items() if 'mmd' in k)
        reg_loss = sum(v for k, v in losses.items() if 'mmd' not in k)
        
        print(f"  MMD损失: {mmd_loss:.4f} ({mmd_loss/total_loss*100:.1f}%)")
        print(f"  正则化损失: {reg_loss:.4f} ({reg_loss/total_loss*100:.1f}%)")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("多K投影方法演示")
    print("=" * 60)
    
    # 创建演示实例
    demo = MultiKProjectionDemo(input_dim=256, num_spaces=10)
    
    # 1. 描述架构
    demo.describe_architecture()
    
    # 2. 演示维度降维
    demo.demonstrate_dimension_reduction()
    
    # 3. 演示多空间融合
    demo.demonstrate_multi_space_fusion()
    
    # 4. 演示曲率优化
    demo.demonstrate_curvature_optimization()
    
    # 5. 演示损失函数
    demo.demonstrate_loss_components()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    print("\n关键创新点:")
    print("1. 使用10个独立的投影空间，每个都有自己的几何特性")
    print("2. 每个空间从高维开始，通过学习逐步降维")
    print("3. 曲率参数可学习，适应数据的内在几何")
    print("4. 智能融合机制自动学习每个空间的重要性")
    print("5. 维度稀疏性促进自动特征选择")


if __name__ == '__main__':
    main()