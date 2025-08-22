# 可学习流形数据凝聚方案

## 概述

本方案实现了一个创新的可学习流形空间方法，用于数据凝聚（Data Condensation）任务。与传统的固定几何空间方法不同，我们的方法允许模型自适应地学习数据的内在几何结构，通过动态调整不同流形空间的曲率来更好地表示和凝聚数据。

## 核心创新点

1. **可学习的流形曲率**：不再固定使用某种特定的几何空间（如欧式、双曲或球面），而是让模型自动学习最适合数据的曲率参数。

2. **多流形空间融合**：同时使用三个流形空间：
   - 一个固定的欧式空间（作为基准）
   - 一个可学习曲率的双曲空间（适合层次结构数据）
   - 一个可学习曲率的球面空间（适合循环或周期性数据）

3. **切空间门控融合**：在共同的切空间中使用门控机制融合不同流形的特征，然后映射回目标流形。

4. **端到端优化**：曲率参数、投影参数和融合权重都可以通过梯度下降端到端优化。

## 系统架构

```
输入图像 → 基础特征提取器 → 欧式特征
                          ↓
                    ┌─────┴─────┐
                    │   投影层   │
                    └─────┬─────┘
           ┌──────────────┼──────────────┐
           ↓              ↓              ↓
      欧式空间      双曲空间(c可学)   球面空间(c可学)
           ↓              ↓              ↓
      切空间映射     切空间映射      切空间映射
           ↓              ↓              ↓
           └──────────────┼──────────────┘
                          ↓
                    门控融合机制
                          ↓
                   指数映射到双曲空间
                          ↓
                      分类/蒸馏
```

## 主要组件

### 1. LearnableManifoldProjection
可学习的流形投影层，支持动态调整曲率。

```python
proj = LearnableManifoldProjection(
    input_dim=128,
    manifold_type='hyperbolic',  # 'hyperbolic', 'spherical', 'euclidean'
    init_curvature=1.0
)
```

**关键特性**：
- 使用对数参数化确保曲率始终为正
- 包含线性投影层适应不同流形的特征表示
- 支持切空间映射（logmap）和指数映射（expmap）

### 2. GatedManifoldFusion
门控机制融合多个流形空间的特征。

```python
fusion = GatedManifoldFusion(
    input_dim=128,
    num_manifolds=3
)
```

**工作原理**：
- 使用注意力机制自动学习不同流形特征的重要性
- 输出归一化的门控权重（和为1）
- 支持任意数量的流形空间

### 3. LearnableManifoldNetwork
完整的可学习流形网络。

```python
model = LearnableManifoldNetwork(
    base_model=base_cnn,
    input_dim=feature_dim,
    num_classes=10,
    hyperbolic_init_c=1.0,
    spherical_init_c=1.0
)
```

### 4. ManifoldDistillationLoss
多流形蒸馏损失函数，包含：
- KL散度损失（logits层面）
- 各流形空间的OT（最优传输）损失
- M3D损失（融合特征层面）
- 门控一致性损失

## 使用方法

### 1. 基本训练流程

```python
# 创建教师模型（标准CNN）
teacher_model = define_model(args, num_classes)

# 创建学生模型（可学习流形模型）
base_student = define_model(args, num_classes)
student_model = create_learnable_manifold_model(
    base_model=base_student,
    input_dim=feature_dim,
    num_classes=num_classes
)

# 定义损失函数
criterion = ManifoldDistillationLoss(temperature=4.0)

# 训练循环
for epoch in range(num_epochs):
    # 教师模型输出
    with torch.no_grad():
        teacher_logits = teacher_model(real_images)
        teacher_info = {...}  # 教师特征信息
    
    # 学生模型输出
    student_logits, student_info = student_model(synthetic_images)
    
    # 计算损失
    loss, loss_dict = criterion(
        student_logits, teacher_logits,
        student_info, teacher_info
    )
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

### 2. 运行数据凝聚

```bash
# 使用可学习流形进行数据凝聚
python condense_learnable_manifold.py \
    --cfg configs/your_config.yaml \
    --dataset cifar10 \
    --ipc 10 \
    --lr_img 0.1 \
    --lr_net 0.01 \
    --temperature 4.0
```

### 3. 测试功能

```bash
# 运行单元测试
python test_learnable_manifold.py
```

## 配置参数

- `hyperbolic_init_c`: 双曲空间初始曲率（默认1.0）
- `spherical_init_c`: 球面空间初始曲率（默认1.0）
- `lr_net`: 网络参数学习率（建议0.01）
- `temperature`: 蒸馏温度（默认4.0）

注意：流形参数（投影层和曲率）建议使用较小的学习率（如主学习率的0.1倍）。

## 理论基础

### 1. 为什么需要可学习的流形？

不同的数据具有不同的内在几何结构：
- **层次数据**：适合双曲空间（负曲率）
- **周期数据**：适合球面空间（正曲率）
- **平坦数据**：适合欧式空间（零曲率）

通过让模型自动学习曲率，我们可以：
- 自适应地发现数据的最佳几何表示
- 避免手动选择流形类型的困难
- 提高模型的表达能力和泛化性

### 2. 切空间融合的优势

在切空间进行融合的好处：
- 切空间是线性的，便于特征组合
- 保持了流形的局部几何性质
- 通过expmap可以映射回任意目标流形

### 3. 门控机制的作用

门控机制允许模型：
- 动态选择最合适的流形表示
- 为不同样本分配不同的流形权重
- 学习数据驱动的几何组合

## 实验建议

1. **初始化策略**：
   - 开始时使用较小的曲率（如0.1-1.0）
   - 让模型逐渐学习合适的曲率值

2. **学习率调度**：
   - 曲率参数使用更小的学习率
   - 可以使用余弦退火或阶梯式衰减

3. **正则化**：
   - 可以添加曲率正则化项，防止曲率过大或过小
   - 门控权重可以添加熵正则化，鼓励探索

4. **监控指标**：
   - 跟踪各流形空间的曲率变化
   - 观察门控权重的分布
   - 记录各损失分量的贡献

## 扩展方向

1. **更多流形类型**：
   - 添加Product空间
   - 实现混合曲率空间
   - 支持黎曼对称空间

2. **自适应流形数量**：
   - 使用稀疏门控选择活跃的流形
   - 动态增加或减少流形空间

3. **层级化流形**：
   - 不同网络层使用不同的流形
   - 学习流形之间的转换

## 引用

如果您使用了这个可学习流形方案，请引用：

```bibtex
@misc{learnable_manifold_condensation,
  title={Learnable Manifold Data Condensation},
  author={Your Name},
  year={2024}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：
- Email: your.email@example.com
- GitHub Issues: https://github.com/yourrepo

---

**注意**：本实现需要安装 `geoopt` 库来处理黎曼流形操作：

```bash
pip install geoopt
```