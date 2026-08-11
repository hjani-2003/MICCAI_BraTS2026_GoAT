from collections.abc import Sequence

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from einops import rearrange

from monai.networks.blocks import PatchEmbed
from monai.utils import look_up_option

from .Transformers_MVM import MERGING_MODE
from .Transformers_MVM.BasicLayer import BasicLayer
from .ConvBlock import ConvBlock

class MambaVision(nn.Module):

    """
        Code adapted from: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,

        window_size: Sequence[int],
        patch_size: Sequence[int],

        depth: int,
        num_heads: int,
        d_state: int,

        drop_rate: float = 0.0,

        norm_layer: type[LayerNorm] = nn.LayerNorm,

        patch_norm: bool = False,
        use_checkpoint: bool = False,

        spatial_dims: int = 3,
        downsample="merging",

    ) -> None:

        super().__init__()
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            spatial_dims=spatial_dims,
        )

        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers1 = ConvBlock(feature_size=embed_dim,
                                 use_checkpoint=use_checkpoint,
                                 norm_layer=norm_layer,
                                 downsample=down_sample_mod,
                                 spatial_dims=spatial_dims)

        self.layers2 = ConvBlock(feature_size=2 * embed_dim, 
                                 use_checkpoint=use_checkpoint,
                                 norm_layer=norm_layer,
                                 downsample=down_sample_mod, 
                                 spatial_dims=spatial_dims)

        self.layers3 = BasicLayer(dim=4 * embed_dim, 
                                  depth=depth, 
                                  num_heads=num_heads, 
                                  d_state=d_state, 
                                  window_size=(7, 7, 7),
                                  norm_layer=norm_layer,
                                  downsample=down_sample_mod,
                                  use_checkpoint=use_checkpoint)

        self.layers4 = BasicLayer(dim=8 * embed_dim,
                                  depth=2*depth, 
                                  num_heads=2*num_heads, 
                                  d_state=2*d_state, 
                                  window_size=(7, 7, 7), 
                                  norm_layer=norm_layer,
                                  downsample=down_sample_mod,
                                  use_checkpoint=use_checkpoint)



    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            ch = int(x_shape[1])
            if len(x_shape) == 5:
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x


    def forward(self, x, normalize=True):

        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)

        x1 = self.layers1(x0.contiguous())
        x1_out = self.proj_out(x1, normalize)

        x2 = self.layers2(x1.contiguous())
        x2_out = self.proj_out(x2, normalize)

        x3 = self.layers3(x2.contiguous())
        x3_out = self.proj_out(x3, normalize)

        x4 = self.layers4(x3.contiguous())
        x4_out = self.proj_out(x4, normalize)

        return [x0_out, x1_out, x2_out, x3_out, x4_out]