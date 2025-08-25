# 多K投影方法 (Multi-K Projection)

这是一个改进的k投影方法实现，支持10个可学习的流形空间，每个空间都有独立的维度和曲率优化能力。

## 主要特性

1. **多空间投影**：支持10个独立的投影空间，每个空间可以是不同类型的流形（双曲、球面、欧式或混合）
2. **维度降维**：每个空间从高维开始，通过可学习的降维网络逐步降到目标维度
3. **曲率学习**：每个空间都有独立的可学习曲率参数
4. **智能融合**：使用注意力机制动态融合多个空间的特征
5. **维度稀疏性**：通过维度权重学习实现自动的特征选择

## 文件结构

- `multi_k_projection.py`: 多K投影网络的核心实现
- `condense_multi_k_projection.py`: 使用多K投影进行数据凝聚的脚本
- `test_multi_k_projection.py`: 测试脚本
- `configs/multi_k_projection_cifar10.yaml`: CIFAR-10的示例配置文件

## 使用方法

### 1. 运行测试

```bash
# 基础功能测试
python test_multi_k_projection.py

# 使用真实模型架构测试
python test_multi_k_projection.py --test_real --dataset cifar10 --num_spaces 10
```

### 2. 运行数据凝聚

```bash
# 使用多K投影进行CIFAR-10数据凝聚
python condense_multi_k_projection.py --cfg configs/multi_k_projection_cifar10.yaml

# 自定义空间数量
python condense_multi_k_projection.py --cfg configs/multi_k_projection_cifar10.yaml --num_spaces 15
```

### 3. 在自己的代码中使用

```python
from multi_k_projection import create_multi_k_projection_model, MultiKProjectionLoss

# 创建基础模型
base_model = YourFeatureExtractor()

# 创建多K投影模型
model = create_multi_k_projection_model(
    base_model=base_model,
    input_dim=feature_dim,
    num_classes=num_classes,
    num_spaces=10  # 使用10个空间
)

# 创建损失函数
criterion = MultiKProjectionLoss(num_spaces=10)

# 前向传播
logits, info = model(images)

# 计算损失
loss, loss_dict = criterion(real_info, syn_info)
```

## 关键参数说明

### MultiKProjectionNetwork参数

- `num_spaces`: 投影空间的数量（默认10）
- `initial_dims`: 每个空间的初始维度列表
- `manifold_types`: 每个空间的流形类型列表
- `init_curvatures`: 每个空间的初始曲率列表

### 训练参数

- `lr_net`: 网络参数的学习率
- `lr_img`: 合成图像的学习率
- `num_spaces`: 使用的投影空间数量

## 架构细节

### 1. 维度降维器 (DimensionReducer)

每个投影空间都有一个独立的维度降维器：
- 从输入维度开始，通过多阶段网络逐步降到目标维度
- 包含可学习的维度权重，用于特征选择
- 使用BatchNorm和ReLU进行正则化

### 2. 多空间投影 (MultiSpaceProjection)

支持四种流形类型：
- **双曲空间 (Hyperbolic)**: 使用Poincaré ball模型
- **球面空间 (Spherical)**: 投影到单位球面
- **欧式空间 (Euclidean)**: 标准欧式空间
- **混合空间 (Mixed)**: 双曲和球面的加权组合

### 3. 空间融合网络

- 使用注意力机制计算每个空间的重要性权重
- 通过独立的融合网络处理每个空间的特征
- 最终拼接并通过深度网络得到统一表示

### 4. 损失函数

包含多个组件：
- **多空间MMD损失**: 在每个投影空间计算真实和合成数据的分布差异
- **空间多样性损失**: 鼓励使用所有投影空间
- **维度稀疏性损失**: 促进自动特征选择
- **曲率正则化**: 保持曲率在合理范围并鼓励多样性
- **最终特征MMD**: 在融合特征上计算分布匹配

## 优势

1. **自适应性**: 模型可以自动学习每个空间的最优维度和曲率
2. **多样性**: 不同的流形类型捕获数据的不同几何特性
3. **可解释性**: 可以分析每个空间的利用率和维度重要性
4. **灵活性**: 易于扩展到更多空间或添加新的流形类型

## 实验建议

1. 对于简单数据集（如MNIST），可以使用较少的空间（5-8个）
2. 对于复杂数据集（如ImageNet），建议使用10-15个空间
3. 初始维度可以根据基础特征维度调整，建议从feature_dim开始逐步降到feature_dim/4
4. 学习率建议：流形参数使用较小的学习率（主学习率的0.1倍）

## 输出文件

训练过程会生成以下文件：
- `data_*.pt`: 不同迭代的合成数据
- `multi_k_params_*.pt`: 保存的模型参数，包括：
  - 每个空间的曲率
  - 空间利用率统计
  - 维度权重
  - 其他配置信息

## 可视化

使用Weights & Biases (wandb)记录训练过程，包括：
- 各个空间的MMD损失
- 曲率变化
- 空间利用率
- 维度稀疏性

## 扩展建议

1. **添加更多流形类型**: 如Grassmann流形、Stiefel流形等
2. **动态空间数量**: 根据数据复杂度自动调整空间数量
3. **层次化投影**: 实现空间之间的层次结构
4. **任务特定优化**: 根据下游任务调整损失函数权重