from typing import Type
from torch.nn import LayerNorm

from torch import nn
import torch.utils.checkpoint as checkpoint

from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(
            self,
            feature_size: int,
            use_checkpoint: bool = False,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            downsample: nn.Module | None = None,
            spatial_dims: int = 3,
    ):
        super().__init__()

        self.feature_size = feature_size
        self.norm_layer = norm_layer(feature_size)

        self.use_checkpoint = use_checkpoint

        self.conv = nn.Conv3d(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(inplace=True)

        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=feature_size, spatial_dims=spatial_dims)


    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.conv, x, use_reentrant=False)
        else:
            x = self.conv(x)

        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm_layer(x)
        x = self.act(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x