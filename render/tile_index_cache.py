# render/tile_index_cache.py
import torch

class TileIndexCache:
    def __init__(self, tile: int = 16):
        self.tile = tile
        self.last_res = None
        self.cache = None

    @torch.no_grad()
    def build_or_update(self, G, cams, res_hw, changed_tiles=None):
        """
        Build on first call; update only changed_tiles afterward for speed.
        """
        H, W = res_hw
        Th, Tw = (H + self.tile - 1)//self.tile, (W + self.tile - 1)//self.tile
        if self.cache is None:
            self.cache = [ [] for _ in range(Th*Tw) ]
            # naive projection: fill cache with gaussian ids per tile (coarse bbox)
            fill_cache(self.cache, G, cams, H, W, self.tile)
        else:
            if changed_tiles is not None:
                update_tiles(self.cache, changed_tiles, G, cams, H, W, self.tile)

    def lookup(self, tile_mask):
        ids = set()
        Th, Tw = tile_mask.shape
        for th in range(Th):
            for tw in range(Tw):
                if tile_mask[th, tw]:
                    ids.update(self.cache[th*Tw + tw])
        return torch.tensor(sorted(ids), dtype=torch.long)
