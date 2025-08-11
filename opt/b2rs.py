# opt/b2rs.py
import torch

@torch.no_grad()
def primal_dual_step(z, residual_per_tile, time_cost_per_tile, B_ms,
                     lam=1e-3, eta_z=0.5, eta_mu=0.5, mu=None):
    if mu is None:
        mu = residual_per_tile.new_tensor(0.0)
    # gradient proxy: tile residual
    g = residual_per_tile
    z = (z - eta_z * (g + lam + mu * time_cost_per_tile)).clamp_(0, 1)
    mu = torch.clamp(mu + eta_mu * ((time_cost_per_tile * z).sum() - B_ms), min=0.0)
    return z, mu
