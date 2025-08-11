# dyn/ot_transport.py
import torch

@torch.no_grad()
def sinkhorn_transport(R_mass, C, eps=0.05, iters=3):
    # R_mass: [P] residual mass per tile (sum=1)
    # C: [P, M] cost (e.g., Mahalanobis tile->gaussian)
    K = torch.exp(-C/eps)
    a = R_mass / (R_mass.sum() + 1e-8)
    b = torch.ones(C.shape[1], device=C.device) / C.shape[1]
    u = torch.ones_like(a); v = torch.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.t() @ u + 1e-8)
    Pi = torch.diag(u) @ K @ torch.diag(v)      # [P, M]
    return Pi

def spawn_cull_from_transport(Pi, null_thresh=0.05, window=5):
    # placeholder: return indices to spawn/cull
    # (실사용에선 누적 통계로 window 처리)
    pass
