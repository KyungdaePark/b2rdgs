# gating/roi_gate.py
import torch
from typing import Tuple, List

@torch.no_grad()
def weighted_residual_lowres(
    frames_lowres: List[torch.Tensor],   # [V, H', W', 3], uint8/float
    pred_lowres:   List[torch.Tensor],   # [V, H', W', 3]
    cams,                                  # camera intrinsics/extrinsics
    use_epipolar: bool = True,
) -> torch.Tensor:
    """
    returns residual map R in [H', W'] (float32)
    if use_epipolar, weight per-view residuals by epipolar consistency & flow-uncertainty
    """
    # L1 residual
    res = [ (f - y).abs().mean(dim=-1) for f, y in zip(frames_lowres, pred_lowres) ]  # each [H', W']
    R = torch.stack(res, dim=0)  # [V, H', W']
    if use_epipolar:
        w = compute_epi_confidence(cams, frames_lowres)  # [V, H', W'], lightweight
        R = (w * R).sum(dim=0) / (w.sum(dim=0) + 1e-6)   # [H', W']
    else:
        R = R.mean(dim=0)
    return R

def top_tiles(R: torch.Tensor, tile: int, top_p: float) -> torch.Tensor:
    """
    returns boolean tile mask [Th, Tw], selecting top_p area by average residual
    """
    H, W = R.shape
    Th, Tw = (H + tile - 1)//tile, (W + tile - 1)//tile
    Rb = R.unfold(0, tile, tile).unfold(1, tile, tile).mean(dim=(2,3))  # [Th, Tw]
    k = max(1, int(Th*Tw*top_p))
    vals, idx = torch.topk(Rb.flatten(), k)
    mask = torch.zeros_like(Rb, dtype=torch.bool)
    mask.view(-1)[idx] = True
    return mask  # [Th, Tw]

def tiles_to_gaussians(tile_mask, tile_index_cache) -> torch.Tensor:
    """
    map active tiles -> gaussian indices (unique)
    """
    return tile_index_cache.lookup(tile_mask)  # 1D LongTensor of ids
