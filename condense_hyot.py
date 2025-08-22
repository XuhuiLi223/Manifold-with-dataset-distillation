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
import geoopt

poincare = geoopt.PoincareBall(c=1.0)

class OTProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim or in_dim
        self.proj = nn.Sequential(
            nn.Linear(self.in_dim, self.proj_dim),
        )

    def forward(self, x):
        return self.proj(x)

class LearnableKManifold(nn.Module):
    """Unified constant-curvature manifold with learnable curvature k.
    k > 0: spherical; k = 0: Euclidean; k < 0: hyperbolic (Poincaré ball model at origin).
    Provides exp/log at the origin.
    """
    def __init__(self, k_init=0.0, k_max=1.0, eps=1e-6):
        super().__init__()
        self.raw_k = nn.Parameter(torch.tensor(float(k_init)))
        self.k_max = float(k_max)
        self.eps = eps

    def curvature(self):
        # Bound curvature to [-k_max, k_max] via tanh
        return self.k_max * torch.tanh(self.raw_k)

    def _safe_norm(self, v):
        return torch.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)

    def exp0(self, v):
        k = self.curvature()
        if torch.isclose(k, torch.tensor(0.0, device=v.device), atol=1e-8):
            return v
        k_val = k.item() if v.numel() > 0 else 0.0
        norm_v = self._safe_norm(v)
        if k_val > 0:
            s = torch.sqrt(k) * norm_v
            scale = torch.tan(s) / s
        else:
            c = -k
            s = torch.sqrt(c) * norm_v
            scale = torch.tanh(s) / s
        return scale * v

    def log0(self, x):
        k = self.curvature()
        if torch.isclose(k, torch.tensor(0.0, device=x.device), atol=1e-8):
            return x
        k_val = k.item() if x.numel() > 0 else 0.0
        norm_x = self._safe_norm(x)
        if k_val > 0:
            s = torch.sqrt(k) * norm_x
            scale = torch.atan(s) / s
        else:
            c = -k
            s = torch.sqrt(c) * norm_x
            # artanh(s) = 0.5 * log((1+s)/(1-s))
            scale = 0.5 * torch.log1p(2 * s / (1 - s + self.eps)) / s
        return scale * x

class MixtureManifoldEncoder(nn.Module):
    """Three-branch encoder from Euclidean features with two learnable-curvature branches and gating in tangent space.
    Branch 0: Euclidean (fixed k=0).
    Branch 1/2: Learnable k via LearnableKManifold.
    After mapping to tangent (origin), a gate fuses the three tangent vectors and is exp-mapped to a target manifold (prefer hyperbolic).
    """
    def __init__(self, input_dim, k1_init=-0.1, k2_init=0.1, k_max=1.0, target_k_init=-0.1):
        super().__init__()
        self.input_dim = input_dim
        # Per-branch linear heads to create tangent vectors
        self.head_euclid = nn.Linear(input_dim, input_dim)
        self.head_k1 = nn.Linear(input_dim, input_dim)
        self.head_k2 = nn.Linear(input_dim, input_dim)
        # Curvature manifolds
        self.manifold_k1 = LearnableKManifold(k_init=k1_init, k_max=k_max)
        self.manifold_k2 = LearnableKManifold(k_init=k2_init, k_max=k_max)
        # Target manifold (prefer hyperbolic): param ensures negative curvature via -softplus
        self.raw_target = nn.Parameter(torch.tensor(float(target_k_init)))
        # Simple gating on norms of tangent vectors -> softmax weights over 3 branches
        self.gate = nn.Linear(3, 3)

        # Initialize heads near identity
        nn.init.eye_(self.head_euclid.weight)
        nn.init.zeros_(self.head_euclid.bias)
        nn.init.eye_(self.head_k1.weight)
        nn.init.zeros_(self.head_k1.bias)
        nn.init.eye_(self.head_k2.weight)
        nn.init.zeros_(self.head_k2.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def target_curvature(self):
        # Negative softplus to bias towards hyperbolic
        return -F.softplus(self.raw_target)

    def exp0_target(self, v):
        kT = self.target_curvature()
        norm_v = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
        if torch.isclose(kT, torch.tensor(0.0, device=v.device), atol=1e-8):
            return v
        if (kT < 0).item():
            c = -kT
            s = torch.sqrt(c) * norm_v
            scale = torch.tanh(s) / s
        else:
            s = torch.sqrt(kT) * norm_v
            scale = torch.tan(s) / s
        return scale * v

    def forward(self, z):
        # Tangent vectors per branch
        v0 = self.head_euclid(z)
        v1 = self.head_k1(z)
        v2 = self.head_k2(z)
        # Norm-based gating
        norms = torch.stack([
            torch.norm(v0, dim=-1),
            torch.norm(v1, dim=-1),
            torch.norm(v2, dim=-1)
        ], dim=-1)  # [B,3]
        w = torch.softmax(self.gate(norms), dim=-1)  # [B,3]
        v_mix = w[:, [0]] * v0 + w[:, [1]] * v1 + w[:, [2]] * v2
        x_mix = self.exp0_target(v_mix)
        return {
            'v_list': [v0, v1, v2],
            'weights': w,
            'v_mix': v_mix,
            'x_mix': x_mix,
            'k_list': [torch.tensor(0.0, device=z.device), self.manifold_k1.curvature(), self.manifold_k2.curvature()],
            'k_target': self.target_curvature(),
        }

class Synthesizer():
    """Condensed data class
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
        # result = test_data(args, loader, val_loader, test_resnet=False, logger=logger)
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
    """Optimize condensed data
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

    # if not args.demo:
    #     synset.test(args, val_loader, logger)
    proj_head = OTProjectionHead(2048).to(device).requires_grad_(True) # 2048 for cifar10, mnist, svhn
    mixture_encoder = MixtureManifoldEncoder(input_dim=2048).to(device)

    # Data distillation
    params = list(synset.parameters()) + list(proj_head.parameters()) + list(mixture_encoder.parameters())
    optim_img = torch.optim.SGD(params, lr=args.lr_img, momentum=args.mom_img)

    n_iter = args.niter
    it_log = 20
    warmup_epoch = 200


    it_test = np.arange(0, n_iter + 1, args.test_it_interval).tolist()

    logger(f"\n M3D: Start condensing with {args.kernel} kernel for {n_iter} iteration")

    best_acc = -1
    eucli_m3d_criterion = M3DLoss(kernel_type=args.kernel)
    hyperbolic_m3d_criterion = M3DLoss(kernel_type="hyperbolic")
    # real_weighted_matrix = torch.ones((nclass, 128))
    # synthetic_weighted_matrix = torch.ones((nclass, 40))
    # real_weight_matrix = torch.nn.Parameter(real_weighted_matrix)
    # synthetic_weight_matrix = torch.nn.Parameter(synthetic_weighted_matrix)

    for it in range(n_iter):

        if it == warmup_epoch:
            for param in proj_head.parameters():
                param.requires_grad = False

        if it % args.ipm == 0:
            model = define_model(args, nclass).to(device)
            width = int(128 * args.width)
            eucli_model = CN.CurvedConvNet(channel=args.nch,
                                     num_classes=nclass,
                                     net_width=width,
                                     net_depth=args.depth,
                                     net_norm=args.norm_type,
                                     im_size=(args.size, args.size)).to(device)
            # Single base embedder; curvature-specific behavior handled by mixture encoder
            model.train()

        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)

        # Update synset
        total_loss_m3d = 0.0
        total_loss_eucli = 0.0
        total_loss_hyper = 0.0
        total_log_eucli = 0.0
        total_log_hyperbolic = 0.0

        for c in range(nclass):

            img, _ = loader_real.class_sample(c)
            img_syn, _ = synset.sample(c, max_size=args.batch_syn_max)

            n = img.shape[0]
            img_aug = aug(torch.cat([img, img_syn]))
            syn_half = (img_aug.shape[0] - n) // 2

            with torch.no_grad():
                feat_base_tg = eucli_model.embed(img_aug[0:n])
            feat_base = eucli_model.embed(img_aug[n:])

            # Three-branch manifold encoding and tangent fusion
            enc_tg = mixture_encoder(feat_base_tg)
            enc_syn = mixture_encoder(feat_base)

            v_tg_list = enc_tg['v_list']
            v_syn_list = enc_syn['v_list']

            # Per-branch OT + M3D losses in tangent (Euclidean) space
            loss_wass_sum = 0.0
            log_loss_sum = 0.0
            loss_m3d_sum = 0.0
            for i in range(3):
                loss_wass_i, log_loss_i = wasserstein_structural_loss(
                    proj_head(v_tg_list[i]), proj_head(v_syn_list[i]))
                loss_m3d_i = eucli_m3d_criterion(v_syn_list[i], v_tg_list[i])
                loss_wass_sum = loss_wass_sum + loss_wass_i
                log_loss_sum = log_loss_sum + log_loss_i
                loss_m3d_sum = loss_m3d_sum + loss_m3d_i

            loss = loss_m3d_sum + args.ot_weight * loss_wass_sum
            loss_total += loss.item()

            total_loss_m3d += loss_m3d_sum.item()
            total_loss_eucli += loss_wass_sum
            total_log_eucli += log_loss_sum

            optim_img.zero_grad()
            loss.backward()
            optim_img.step()

        avg_loss_m3d = total_loss_m3d / nclass
        avg_loss_eucli = total_loss_eucli / nclass
        # avg_loss_hyper = total_loss_hyper / nclass
        # avg_log_hyper = total_log_hyperbolic / nclass
        avg_log_eucli = total_log_eucli / nclass
        wandb.log({
            "iteration": it,
            "loss_m3d": avg_loss_m3d,
            "loss_wasserstein_eucli": avg_loss_eucli,
            # "loss_wasserstein_hyper": avg_loss_hyper,
            "log_loss_eucli": avg_log_eucli,
            # "log_loss_hyper": avg_log_hyper,
        })

        if args.kernel == 'gaussian':
            loss_total *= 1000  # to moniter the loss value
        elif args.kernel == 'linear':
            loss_total *= 100
        else:
            pass  # todo

        # Logging
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total / nclass:.2f}")

        if (it + 1) in it_test:
            save_img(os.path.join(args.save_dir, f'img{it + 1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=args.dataset)

            # It is okay to clamp data to [0, 1] at here.
            # synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)

            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, 'data_{}.pt'.format(it + 1)))
            logger("img and data saved!")

            if not args.demo:

                conv_result = synset.test(args, val_loader, logger)
                if conv_result > best_acc:
                    best_acc = conv_result
                    torch.save(
                        [synset.data.detach().cpu(), synset.targets.cpu()],
                        os.path.join(args.save_dir, 'data_best.pt'))

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
    parser.add_argument(
        "--softlabel", default=False,
        dest="softlabel",
        help="Use the softlabel to evaluate the dataset",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="The temperature for KLdiv"
    )
    parser.add_argument("--ot_weight", type=int, default=0.001) #cifar10:1, svhn:0.001
    parser.add_argument("--ot_weight_hyperbolic", type=int, default=0.5)

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    for key, value in cfg.items():
        arg_name = '--' + key
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()

    mode = "online"
    wandb.init(
        # project=f'One-shot_FL_{args.dataset}_test',
        project=f'Manifold_{args.dataset}_new_test',
        name=f'Model_{args.net_type}_Method_wasserstein_structural_loss_with_product_space', #origin
        mode=mode,
    )

    cudnn.benchmark = True
    print("CUDNN STATUS: {}".format(cudnn.enabled))
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    DATANAME = args.dataset if args.dataset != 'imagenet' else "{}{}".format(args.dataset, args.nclass)
    args.save_dir = os.path.join(args.results_path, DATANAME, "IPC" + str(args.ipc))
    os.makedirs(args.save_dir, exist_ok=True)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")

    with open(os.path.join(args.save_dir, 'args.log'), 'w') as f:
        json.dump(args.__dict__, f, indent=3)

    condense(args, logger)

    # # 每个image能表示成其他图像的linear combination，从而有一组基，从而把图像用矩阵表示成base的乘积，生成一批product space的数据，把三个embedding拼起来，然后判断是否能比单一空间更好，说明是否product space是有效的。
    # #
    # # deeplearnning的基础假设，如果低维流形不变，那么我们的方法是否有很好的鲁棒性，因为高维数据都具有低维流形，如果能够学习到低维流形，那么是否有更好的效果。
    # 现有的迁移方式是optimal transport, 目前使用多个空间，是一种很浅显的流形刻画，如何逼近底层流形
    # 多个空间的mix curvature，多个空间的product space，ot在manifold的迁移方式