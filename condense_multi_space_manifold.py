import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
from data import TensorDataset, ImageFolder, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from model_dist import define_model
from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
import sys
from m3dloss import *
import models.convnet as CN
from evaluate_synset import evaluate_syn_data
import wandb
from learnable_manifold_multi_space import (
    MultiSpaceLearnableManifoldNetwork, 
    MultiSpaceManifoldMatchingLoss, 
    create_multi_space_manifold_model
)
from geoopt.optim import RiemannianAdam


class MultiSpaceSynthesizer():
    """多空间流形数据合成器"""

    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print(f"\n定义多空间合成数据: {self.data.shape}")

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """合成数据初始化"""
        if init_type == 'random':
            print("随机初始化合成数据集")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("混合初始化合成数据集")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor ** 2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                        w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            print("噪声初始化合成数据集")
            pass

    def parameters(self):
        return [self.data]

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode(self, data, target, bound=128):
        """解码数据"""
        if self.factor > 1:
            data = self.resize(data)

        data = torch.clamp(data, min=0., max=1.)
        if self.decode_type == 'bound':
            data = torch.clamp(data, min=0., max=1.)

        return data, target

    def sample(self, c, max_size=128):
        """采样指定类别的数据"""
        idx = self.cls_idx[c]
        if len(idx) <= max_size:
            return self.data[idx], self.targets[idx]
        else:
            idx = np.random.choice(idx, max_size, replace=False)
            return self.data[idx], self.targets[idx]

    def loader(self, args, augment=True):
        """创建数据加载器"""
        if augment and args.dsa:
            aug, args.dsa_param = DiffAug(args.dsa_strategy, seed=args.dsa_seed, param=args.dsa_param)
            print('DSA augmentation strategy: ', args.dsa_strategy)
            print('DSA augmentation parameters: ', args.dsa_param)
        else:
            aug = None

        loader = MultiEpochsDataLoader(TensorDataset(self.data, self.targets, aug), 
                                       batch_size=args.batch_syn, 
                                       shuffle=True, 
                                       num_workers=0, 
                                       persistent_workers=False)
        return loader

    def test(self, args, val_loader, logger, bench=True):
        """测试合成数据"""
        loader = self.loader(args, augment=False)
        return test_data(args, loader, val_loader, test_resnet=False, logger=logger)


def condense_multi_space(args, logger):
    """多空间流形数据凝聚主函数"""
    
    # 设置设备和种子
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 数据加载
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transform_cifar(args, train=True))
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_cifar(args, train=False))
        nclass = 10
        nch, hs, ws = 3, 32, 32
        mean, std = MEANS['cifar10'], STDS['cifar10']
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir, train=True, transform=transform_cifar(args, train=True))
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_cifar(args, train=False))
        nclass = 100
        nch, hs, ws = 3, 32, 32
        mean, std = MEANS['cifar100'], STDS['cifar100']
    else:
        raise NotImplementedError(f"数据集 {args.dataset} 尚未实现")

    # 创建数据加载器
    train_loader = ClassMemDataLoader(train_dataset, batch_size=args.batch_train)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_train, shuffle=False, num_workers=4)

    # 创建基础模型
    base_model = define_model(args, nclass)
    
    # 获取特征维度
    with torch.no_grad():
        sample_data = torch.randn(1, nch, hs, ws).to(args.device)
        sample_features = base_model.embed(sample_data)
        feature_dim = sample_features.shape[1]
    
    print(f"特征维度: {feature_dim}")
    
    # 创建多空间流形模型
    model = create_multi_space_manifold_model(
        base_model=base_model,
        input_dim=feature_dim,
        num_classes=nclass,
        num_spaces=args.num_spaces,  # 默认10个空间
        target_dim_range=(args.min_dim, args.max_dim)  # 维度范围
    ).to(args.device)
    
    print(f"创建了 {args.num_spaces} 个流形空间，维度范围: [{args.min_dim}, {args.max_dim}]")
    
    # 创建合成数据
    synthesizer = MultiSpaceSynthesizer(args, nclass, nch, hs, ws, args.device)
    synthesizer.init(train_loader, init_type=args.init)
    
    # 定义损失函数
    criterion = MultiSpaceManifoldMatchingLoss(num_spaces=args.num_spaces).to(args.device)
    
    # 优化器设置
    # 分别为不同类型的参数设置不同的学习率
    img_params = synthesizer.parameters()
    
    # 网络参数分组
    manifold_params = []  # 流形相关参数（曲率、投影等）
    regular_params = []   # 常规参数
    
    for name, param in model.named_parameters():
        if 'log_curvature' in name or 'dim_logit' in name or 'projection' in name:
            manifold_params.append(param)
        else:
            regular_params.append(param)
    
    # 创建优化器
    optimizer_img = torch.optim.SGD(img_params, lr=args.lr_img, momentum=args.mom_img)
    optimizer_net = torch.optim.Adam(regular_params, lr=args.lr_net)
    optimizer_manifold = torch.optim.Adam(manifold_params, lr=args.lr_net * 0.1)  # 流形参数用较小学习率
    
    # 学习率调度器
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.epochs)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=args.epochs)
    scheduler_manifold = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_manifold, T_max=args.epochs)
    
    # 训练循环
    best_acc = 0.0
    loss_history = []
    
    for epoch in range(args.epochs):
        model.train()
        
        # 获取真实数据批次
        real_images, real_labels = train_loader.sample()
        real_images, real_labels = real_images.to(args.device), real_labels.to(args.device)
        
        # 获取合成数据
        syn_images = synthesizer.data
        syn_labels = synthesizer.targets
        
        # 子采样以匹配批次大小
        if real_images.shape[0] > syn_images.shape[0]:
            indices = torch.randperm(real_images.shape[0])[:syn_images.shape[0]]
            real_images = real_images[indices]
            real_labels = real_labels[indices]
        elif syn_images.shape[0] > real_images.shape[0]:
            indices = torch.randperm(syn_images.shape[0])[:real_images.shape[0]]
            syn_images = syn_images[indices]
            syn_labels = syn_labels[indices]
        
        # 前向传播
        real_logits, real_info = model(real_images, epoch=epoch)
        syn_logits, syn_info = model(syn_images, epoch=epoch)
        
        # 计算损失
        loss, loss_dict = criterion(real_logits, syn_logits, real_info, syn_info)
        
        # 反向传播
        optimizer_img.zero_grad()
        optimizer_net.zero_grad()
        optimizer_manifold.zero_grad()
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(img_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer_img.step()
        optimizer_net.step()
        optimizer_manifold.step()
        
        # 更新学习率
        scheduler_img.step()
        scheduler_net.step()
        scheduler_manifold.step()
        
        loss_history.append(loss.item())
        
        # 日志记录
        if epoch % 10 == 0:
            # 记录当前各空间的维度和曲率
            current_dims = syn_info['current_dims']
            curvatures = syn_info['curvatures']
            fusion_gates = syn_info['fusion_gates'].mean(dim=0)
            
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"总损失: {loss.item():.4f}")
            print(f"当前维度: {current_dims}")
            print(f"曲率值: {[f'{c:.3f}' if isinstance(c, torch.Tensor) else f'{c:.3f}' for c in curvatures]}")
            print(f"融合权重: {[f'{g:.3f}' for g in fusion_gates.cpu().numpy()]}")
            
            for key, value in loss_dict.items():
                print(f"  {key}: {value:.4f}")
            
            # 记录到wandb（如果启用）
            if hasattr(args, 'wandb') and args.wandb:
                log_dict = {
                    'epoch': epoch,
                    'total_loss': loss.item(),
                    'avg_dimension': np.mean(current_dims),
                    'avg_curvature': np.mean([c.item() if isinstance(c, torch.Tensor) else c for c in curvatures]),
                }
                log_dict.update(loss_dict)
                wandb.log(log_dict)
        
        # 定期测试
        if epoch % 50 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                acc = synthesizer.test(args, val_loader, logger, bench=False)
                print(f"测试准确率: {acc:.2f}%")
                
                if acc > best_acc:
                    best_acc = acc
                    # 保存最佳模型
                    save_path = os.path.join(args.save_dir, 'best_multi_space_model.pth')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'synthesizer_data': synthesizer.data,
                        'epoch': epoch,
                        'best_acc': best_acc,
                        'args': args
                    }, save_path)
                    print(f"保存最佳模型到: {save_path}")
            
            model.train()
    
    # 最终测试
    model.eval()
    with torch.no_grad():
        final_acc = synthesizer.test(args, val_loader, logger, bench=True)
        print(f"\n最终测试准确率: {final_acc:.2f}%")
        print(f"最佳测试准确率: {best_acc:.2f}%")
    
    # 保存最终结果
    final_save_path = os.path.join(args.save_dir, 'final_multi_space_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'synthesizer_data': synthesizer.data,
        'final_acc': final_acc,
        'best_acc': best_acc,
        'loss_history': loss_history,
        'args': args
    }, final_save_path)
    
    # 保存合成数据集
    syn_data_path = os.path.join(args.save_dir, 'synthetic_dataset.pth')
    torch.save({
        'data': synthesizer.data.cpu(),
        'targets': synthesizer.targets.cpu(),
        'nclass': nclass,
        'shape': (nch, hs, ws)
    }, syn_data_path)
    
    print(f"训练完成！最终模型保存到: {final_save_path}")
    print(f"合成数据保存到: {syn_data_path}")
    
    return final_acc, best_acc


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多空间流形数据凝聚')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results_multi_space')
    parser.add_argument('--ipc', type=int, default=10, help='每个类别的合成图像数量')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    
    # 多空间参数
    parser.add_argument('--num_spaces', type=int, default=10, help='流形空间数量')
    parser.add_argument('--min_dim', type=int, default=32, help='最小维度')
    parser.add_argument('--max_dim', type=int, default=256, help='最大维度')
    
    # 学习率参数
    parser.add_argument('--lr_img', type=float, default=0.1, help='合成图像学习率')
    parser.add_argument('--lr_net', type=float, default=0.01, help='网络参数学习率')
    parser.add_argument('--mom_img', type=float, default=0.9, help='合成图像动量')
    
    # 批次大小
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--batch_syn', type=int, default=256)
    
    # 初始化和增强
    parser.add_argument('--init', type=str, default='noise', choices=['noise', 'random', 'mix'])
    parser.add_argument('--dsa', action='store_true', help='启用数据增强')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--dsa_seed', type=int, default=0)
    parser.add_argument('--dsa_param', type=float, default=0.5)
    
    # 模型参数
    parser.add_argument('--model', type=str, default='convnet', help='模型类型')
    parser.add_argument('--depth', type=int, default=3, help='网络深度')
    parser.add_argument('--width', type=int, default=128, help='网络宽度')
    parser.add_argument('--norm', type=str, default='instancenorm', help='归一化类型')
    parser.add_argument('--pooling', type=str, default='avgpooling', help='池化类型')
    
    # 其他参数
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--decode_type', type=str, default='single', choices=['single', 'multi', 'bound'])
    parser.add_argument('--wandb', action='store_true', help='启用wandb日志')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化wandb（如果启用）
    if args.wandb:
        wandb.init(
            project="multi-space-manifold-condensation",
            config=args,
            name=f"{args.dataset}_spaces{args.num_spaces}_ipc{args.ipc}"
        )
    
    # 创建日志记录器
    logger = utils.Logger(args.save_dir)
    logger.print_args(args)
    
    # 运行训练
    final_acc, best_acc = condense_multi_space(args, logger)
    
    # 关闭wandb
    if args.wandb:
        wandb.finish()
    
    print(f"\n=== 训练完成 ===")
    print(f"最终准确率: {final_acc:.2f}%")
    print(f"最佳准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()