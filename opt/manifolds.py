# opt/manifolds.py
import torch

def skew(v):  # v: [...,3]
    x,y,z = v.unbind(-1)
    O = torch.zeros_like(v[..., :1]).expand(v.shape[:-1] + (3,3))
    K = O.clone()
    K[...,0,1], K[...,0,2] = -z,  y
    K[...,1,0], K[...,1,2] =  z, -x
    K[...,2,0], K[...,2,1] = -y,  x
    return K

def expm_so3(K):  # Rodrigues with safe small-angle
    # K: [...,3,3] skew-sym
    t = torch.stack([K[...,2,1], K[...,0,2], K[...,1,0]], dim=-1).norm(dim=-1, keepdim=True) + 1e-8
    I = torch.eye(3, device=K.device, dtype=K.dtype).expand_as(K)
    A = (torch.sin(t)/t)[...,None]
    B = ((1 - torch.cos(t))/(t*t))[...,None]
    return I + A*K + B*(K@K)

@torch.no_grad()
def so3_update(R, delta):
    return expm_so3(skew(delta)) @ R

@torch.no_grad()
def spd_update(Sigma, dS):
    L, Q = torch.linalg.eigh(Sigma)                       # Sigma = Q diag(L) Q^T
    logS = Q @ torch.diag_embed(torch.log(torch.clamp(L, min=1e-8))) @ Q.transpose(-1,-2)
    logS = logS + dS
    L2, Q2 = torch.linalg.eigh(logS)
    return Q2 @ torch.diag_embed(torch.exp(L2)) @ Q2.transpose(-1,-2)

def barrier_sigma(Sigma, beta=1e-3):
    # add to loss: -logdet(Sigma) + beta * ||Sigma||_F^2
    sign, logdet = torch.slogdet(Sigma)
    return -(logdet) + beta * (Sigma*Sigma).sum()
