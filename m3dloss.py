import torch
from torch import nn
import ot
import numpy as np
from geomloss import SamplesLoss
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class PoliKernel(nn.Module):

    def __init__(self, constant_term=1, degree=2):
        super().__init__()
        self.constant_term = constant_term
        self.degree = degree


    def forward(self, X):
        K = (torch.matmul(X, X.t()) + self.constant_term) ** self.degree
        return K

class LinearKernel(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, X):
        K = torch.matmul(X, X.t())
        return K

class LaplaceKernel(nn.Module):

    def __init__(self):
        super().__init__()
        self.gammas = torch.FloatTensor([0.1, 1, 5]).cuda()


    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] * (self.gammas)[:, None, None]).sum(dim=0)

class HyperbolicKernel(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.eps = 1e-5

    def poincare_distance(self, x, y):
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_max(1 - self.eps)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True).clamp_max(1 - self.eps)

        x_exp = x.unsqueeze(1)  # [N, 1, D]
        y_exp = y.unsqueeze(0)  # [1, M, D]
        diff_sq = torch.sum((x_exp - y_exp) ** 2, dim=-1)  # [N, M]

        denom = (1 - x_norm_sq) @ (1 - y_norm_sq).T  # [N, M]
        z = 1 + 2 * diff_sq / denom.clamp_min(self.eps)

        return torch.acosh(z.clamp_min(1 + self.eps))  # [N, M]

    def forward(self, X):
        # X: [N + M, D]
        return torch.exp(-self.poincare_distance(X, X) / (2 * self.sigma ** 2))


class M3DLoss(nn.Module):

    def __init__(self, kernel_type):
        super().__init__()
        if kernel_type == 'gaussian':
            self.kernel = RBF()
        elif kernel_type == 'linear':
            self.kernel = LinearKernel()
        elif kernel_type == 'polinominal':
            self.kernel = PoliKernel()
        elif kernel_type == 'laplace':
            self.kernel = LaplaceKernel()
        elif kernel_type == 'hyperbolic':
            self.kernel = HyperbolicKernel()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class OTProjectionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4)
        )
    def forward(self, x):
        self.proj(x)
        x = torch.clamp(x, -1e4, 1e4)
        return x
def compute_ot_distance(feat_src, feat_tgt, reg=0.1):
    # feat_src, feat_tgt: shape [B, D]
    X_src = feat_src.detach().cpu().numpy()
    X_tgt = feat_tgt.detach().cpu().numpy()

    n = X_src.shape[0]
    m = X_tgt.shape[0]

    a = ot.unif(n)
    b = ot.unif(m)

    cost_matrix = ot.dist(X_src, X_tgt, metric='euclidean')  # [n, m]
    cost_matrix /= cost_matrix.max()
    T = ot.sinkhorn(a, b, cost_matrix, reg=reg)
    ot_loss = np.sum(T * cost_matrix)
    return torch.tensor(ot_loss, dtype=torch.float32, device=feat_src.device)

def hyperbolic_distance_poincare(x, y, eps=1e-5):
    """
    x: [N, D], y: [M, D] in Poincaré ball (norm < 1)
    Return: [N, M] matrix of hyperbolic distances
    """
    x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True).clamp_max(1 - eps)
    y_norm_sq = torch.sum(y**2, dim=-1, keepdim=True).clamp_max(1 - eps)

    x_exp = x.unsqueeze(1)  # [N, 1, D]
    y_exp = y.unsqueeze(0)  # [1, M, D]

    diff_sq = torch.sum((x_exp - y_exp)**2, dim=-1)  # [N, M]

    denom = (1 - x_norm_sq) @ (1 - y_norm_sq).T  # [N, M]
    z = 1 + 2 * diff_sq / denom.clamp_min(eps)

    return torch.acosh(z.clamp_min(1 + eps))  # ensure domain valid for acosh

def compute_ot_hyperbolic(feat_src, feat_tgt, reg=0.1):
    x = feat_src.detach()
    y = feat_tgt.detach()

    cost_matrix = hyperbolic_distance_poincare(x, y)  # [N, M]

    n, m = cost_matrix.shape
    a = ot.unif(n)
    b = ot.unif(m)

    cost_np = cost_matrix.cpu().numpy()
    cost_np /= cost_np.max()
    T = ot.sinkhorn(a, b, cost_np, reg=reg)
    ot_loss = np.sum(T * cost_np)

    return torch.tensor(ot_loss, dtype=torch.float32, device=feat_src.device)


def euclidean_cdist(x, y):
    return torch.cdist(x, y, p=2)

# 双曲距离函数
def hyperbolic_cdist(x, y, manifold):
    return manifold.dist(x.unsqueeze(1), y.unsqueeze(0))

# Step 1: Compute transport plan (differentiable version)
def compute_transport_plan(mu_feats, nu_feats, reg, is_hyperbolic=False, manifold=None, verbose=True):
    if is_hyperbolic:
        assert manifold is not None, "Hyperbolic mode requires a manifold object"
        cost_matrix = hyperbolic_cdist(mu_feats, nu_feats, manifold)
    else:
        cost_matrix = euclidean_cdist(mu_feats, nu_feats)

    # Standardize cost matrix
    cost_mean = cost_matrix.mean()
    cost_std = cost_matrix.std()
    cost_matrix = (cost_matrix - cost_mean) / (cost_std + 1e-8)

    # Clamp cost values to prevent extreme values in exp
    scaled_cost = -cost_matrix / reg
    scaled_cost = torch.clamp(scaled_cost, min=-50.0, max=50.0)

    # log-sum-exp trick for better numerical stability
    max_val, _ = scaled_cost.max(dim=1, keepdim=True)
    T = torch.exp(scaled_cost - max_val)
    row_sum = T.sum(dim=1, keepdim=True)

    # Avoid division by zero
    row_sum[row_sum == 0] = 1e-6
    T = T / row_sum

    # Safety check: replace NaNs with 0
    T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    return T


# Step 2: Log-map projection using OT plan

def log_map_projection_from_transport(mu_feats, nu_feats, transport_plan, is_hyperbolic=False, manifold=None):
    nu_proj = transport_plan @ nu_feats
    if is_hyperbolic:
        assert manifold is not None, "Hyperbolic mode requires a manifold object"
        log_vectors = manifold.logmap(mu_feats, nu_proj)
    else:
        log_vectors = nu_proj - mu_feats
    return log_vectors

# Step 3: Log-map structure loss

def log_map_structure_loss(mu_feats, nu_feats, reg, is_hyperbolic=False, manifold=None):
    T = compute_transport_plan(mu_feats, nu_feats, reg, is_hyperbolic, manifold)
    log_vectors = log_map_projection_from_transport(mu_feats, nu_feats, T, is_hyperbolic, manifold)
    loss = torch.mean(torch.norm(log_vectors, dim=1) ** 2)
    return loss


# Step 4: Total Wasserstein + log-map loss
from geomloss import SamplesLoss

def wasserstein_structural_loss(mu_feats, nu_feats, reg=0.5, lambda_log=1.0, is_hyperbolic=False, manifold=None):
    if torch.isnan(mu_feats).any():
        print("[Warning] NaN detected in mu_feats, replacing with 0")
        mu_feats = torch.nan_to_num(mu_feats, nan=0.0)
    if torch.isnan(nu_feats).any():
        print("[Warning] NaN detected in nu_feats, replacing with 0")
        nu_feats = torch.nan_to_num(nu_feats, nan=0.0)

    if mu_feats.std() < 1e-6:
        print("[Warning] Mu Feature variance too small, skipping OT loss")
        return torch.tensor(0.0, device=mu_feats.device), 0.0

    if nu_feats.std() < 1e-6:
        print("[Warning] Nu Feature variance too small")
        return torch.tensor(0.0, device=mu_feats.device), 0.0

    if is_hyperbolic:
        # 使用双曲距离的样本 OT loss（建议自定义或使用 cost_matrix + log loss）
        ot_loss = torch.tensor(0.0, device=mu_feats.device)  # 也可以自己实现 OT loss in Hyperbolic
    else:
        loss_fn_geom = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9, reach=1e-3, debias=True)
        ot_loss = loss_fn_geom(nu_feats, mu_feats)

    log_loss = log_map_structure_loss(mu_feats, nu_feats, reg, is_hyperbolic, manifold)
    if torch.isnan(ot_loss):
        print("ot_loss is NaN")
    if torch.isnan(log_loss):
        print("log_map loss is NaN")
    return ot_loss + lambda_log * log_loss, log_loss


# from geomloss import SamplesLoss
#
# # Step 1: Sinkhorn-based Wasserstein loss (fully differentiable)
# loss_fn_sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.01)  # blur ≈ reg
#
# def log_map_vector_loss(mu_feats, nu_feats):
#     """
#     Log-map loss: measure structure shift in Wasserstein tangent space
#     Approximate using mean vector shift (can be extended)
#     """
#     mu_mean = mu_feats.mean(dim=0)
#     nu_mean = nu_feats.mean(dim=0)
#     log_vector = nu_mean - mu_mean
#     return torch.norm(log_vector, p=2)
#
# def wasserstein_geomloss_structural_loss(mu_feats, nu_feats, lambda_log=1.0):
#     """
#     Combines differentiable OT loss (via geomloss) and structure-preserving log-map loss
#     Both are torch-based, gradient-compatible
#     """
#     ot_loss = loss_fn_sinkhorn(nu_feats, mu_feats)  # Sinkhorn OT
#     log_loss = log_map_vector_loss(mu_feats, nu_feats)
#     return ot_loss + lambda_log * log_loss, ot_loss.detach(), log_loss.detach()
