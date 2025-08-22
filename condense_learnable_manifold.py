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
from learnable_manifold import LearnableManifoldNetwork, ManifoldMatchingLoss, create_learnable_manifold_model
from geoopt.optim import RiemannianAdam


class Synthesizer():
    """Condensed data class with learnable manifold support
    """

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

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
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
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor ** 2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor ** 2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):

        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet' or args.dataset == 'tiny':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        result = evaluate_syn_data(args, loader, val_loader, logger=logger)
        return result


def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor(), download=True)

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'SVHN'),
                                      split='train',
                                      transform=transforms.ToTensor(), download=True)
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'SVHN'),
                                    split='test',
                                    transform=transform_test, download=True)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor(), download=True)

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test, download=True)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor(), download=True)

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'imagenet' or args.dataset == 'tiny':
        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')

        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=-1,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=-1,
                                  seed=args.dseed,
                                  load_memory=False)

    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    return aug_batch


def condense(args, logger, device='cuda'):
    """Optimize condensed data with learnable manifold
    """
    # Define real dataset and loader
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # Define syn dataset
    synset = Synthesizer(args, nclass, nch, hs, ws)
    synset.init(loader_real, init_type=args.init)
    save_img(os.path.join(args.save_dir, 'init.png'),
             synset.data,
             unnormalize=False,
             dataname=args.dataset)

    # Define augmentation function
    aug = diffaug(args)

    torch.save(
        [synset.data.detach().cpu(), synset.targets.cpu()],
        os.path.join(args.save_dir, 'data_0.pt'))

    # 创建可学习流形模型
    base_model = define_model(args, nclass).to(device)
    
    # 计算特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, nch, hs, ws).to(device)
        dummy_features = base_model.embed(dummy_input)
        feature_dim = dummy_features.shape[1]
    
    # 创建可学习流形网络
    model = create_learnable_manifold_model(
        base_model=base_model,
        input_dim=feature_dim,
        num_classes=nclass
    ).to(device)
    
    # 定义损失函数
    criterion = ManifoldMatchingLoss()
    
    # 数据优化器
    # 包含：合成数据、模型参数（包括流形投影参数）
    params = list(synset.parameters())
    
    # 模型参数
    euclidean_params = []
    manifold_params = []
    
    for name, param in model.named_parameters():
        if 'proj' in name or 'gate' in name:
            manifold_params.append(param)
        else:
            euclidean_params.append(param)
    
    # 使用不同的学习率
    optim_img = torch.optim.SGD([
        {'params': synset.parameters(), 'lr': args.lr_img},
        {'params': euclidean_params, 'lr': args.lr_net},
        {'params': manifold_params, 'lr': args.lr_net * 0.1}  # 流形参数使用较小学习率
    ], momentum=args.mom_img)

    n_iter = args.niter
    it_log = 20

    it_test = np.arange(0, n_iter + 1, args.test_it_interval).tolist()

    logger(f"\n Learnable Manifold Condensation: Start condensing for {n_iter} iterations")

    best_acc = -1
    
    for it in range(n_iter):
        
        # 每隔一定步数重新初始化模型
        if it % args.ipm == 0:
            base_model = define_model(args, nclass).to(device)
            model = create_learnable_manifold_model(
                base_model=base_model,
                input_dim=feature_dim,
                num_classes=nclass
            ).to(device)
            model.train()

        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)

        # Update synset
        total_losses = {
            'total': 0.0,
            'eucl_mmd': 0.0,
            'hyp_mmd': 0.0,
            'sph_mmd': 0.0,
            'fused_mmd': 0.0,
            'gate_loss': 0.0,
            'curvature_reg': 0.0,
            'entropy_reg': 0.0
        }

        for c in range(nclass):
            img_real, _ = loader_real.class_sample(c)
            img_syn, _ = synset.sample(c, max_size=args.batch_syn_max)

            n = img_real.shape[0]
            img_aug = aug(torch.cat([img_real, img_syn]))

            # 真实数据特征
            _, real_info = model(img_aug[:n])
            
            # 合成数据特征
            _, syn_info = model(img_aug[n:])
            
            # 计算损失
            loss, loss_dict = criterion(
                None, None,  # 不使用logits
                real_info, syn_info
            )
            
            loss_total += loss.item()
            
            # 累加各项损失
            for key in loss_dict:
                if key in total_losses:
                    total_losses[key] += loss_dict[key]

            optim_img.zero_grad()
            loss.backward()
            optim_img.step()

        # 计算平均损失
        for key in total_losses:
            total_losses[key] /= nclass

        # 记录到wandb
        wandb.log({
            "iteration": it,
            "loss/total": loss_total / nclass,
            "loss/eucl_mmd": total_losses['eucl_mmd'],
            "loss/hyp_mmd": total_losses['hyp_mmd'],
            "loss/sph_mmd": total_losses['sph_mmd'],
            "loss/fused_mmd": total_losses['fused_mmd'],
            "loss/gate": total_losses['gate_loss'],
            "loss/curvature_reg": total_losses['curvature_reg'],
            "loss/entropy_reg": total_losses['entropy_reg'],
            "curvature/hyperbolic": model.hyperbolic_proj.curvature.item(),
            "curvature/spherical": model.spherical_proj.curvature.item()
        })

        # Logging
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total / nclass:.4f}, "
                f"hyp_c: {model.hyperbolic_proj.curvature.item():.3f}, "
                f"sph_c: {model.spherical_proj.curvature.item():.3f}"
            )

        if (it + 1) in it_test:
            save_img(os.path.join(args.save_dir, f'img{it + 1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=args.dataset)

            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, 'data_{}.pt'.format(it + 1)))
            
            # 保存学到的曲率参数
            torch.save({
                'hyperbolic_curvature': model.hyperbolic_proj.curvature.item(),
                'spherical_curvature': model.spherical_proj.curvature.item(),
                'iteration': it + 1
            }, os.path.join(args.save_dir, f'curvatures_{it + 1}.pt'))
            
            logger("img, data and curvatures saved!")

            if not args.demo:
                conv_result = synset.test(args, val_loader, logger)
                if conv_result > best_acc:
                    best_acc = conv_result
                    torch.save(
                        [synset.data.detach().cpu(), synset.targets.cpu()],
                        os.path.join(args.save_dir, 'data_best.pt'))
                    torch.save({
                        'hyperbolic_curvature': model.hyperbolic_proj.curvature.item(),
                        'spherical_curvature': model.spherical_proj.curvature.item(),
                        'iteration': it + 1
                    }, os.path.join(args.save_dir, 'curvatures_best.pt'))

                logger("->->->->->->->->->->->->-> Best Result: {:.1f}".format(best_acc))
                wandb.log({"Best Result": best_acc})


if __name__ == '__main__':
    from misc.utils import Logger
    import torch.backends.cudnn as cudnn
    import json
    import argparse
    from misc.cfg import CFG as cfg

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--pretrain_dir", type=str, default="checkpoints")
    parser.add_argument("--demo", action='store_true', help='for debugging, do not save results')
    parser.add_argument("--lr_net", type=float, default=0.01, help="learning rate for network parameters")

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    for key, value in cfg.items():
        arg_name = '--' + key
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()

    mode = "online"
    wandb.init(
        project=f'LearnableManifold_{args.dataset}',
        name=f'Model_{args.net_type}_Method_learnable_manifold_condensation_no_teacher',
        mode=mode,
    )

    cudnn.benchmark = True
    print("CUDNN STATUS: {}".format(cudnn.enabled))
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    DATANAME = args.dataset if args.dataset != 'imagenet' else "{}{}".format(args.dataset, args.nclass)
    args.save_dir = os.path.join(args.results_path, DATANAME, "IPC" + str(args.ipc), "learnable_manifold")
    os.makedirs(args.save_dir, exist_ok=True)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")

    with open(os.path.join(args.save_dir, 'args.log'), 'w') as f:
        json.dump(args.__dict__, f, indent=3)

    condense(args, logger)