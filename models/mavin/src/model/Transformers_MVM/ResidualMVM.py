from torch import nn
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from monai.networks.blocks.mlp import MLPBlock as Mlp
from einops import rearrange

from .MambaVisionMixer import MambaVisionMixer



class ResidualMVM(nn.Module):
    def __init__(
        self,
        dim,
        d_state,
        d_conv=4,
        expand=4,
        mlp_ratio=4,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        act_layer="GELU",
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.mixer = MambaVisionMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            hidden_size=dim,
            mlp_dim=mlp_hidden_dim,
            act=act_layer,
        )

        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)

    def _forward_mixer(self, x):
        _, d, h, w, _ = x.shape
        x = rearrange(x, 'b d h w c -> b (d h w) c')
        x = self.mixer(self.norm_1(x))
        x = rearrange(x, 'b (d h w) c -> b d h w c', d=d, h=h, w=w)
        return x

    def _forward_mlp(self, x):
        return self.mlp(self.norm_2(x))

    def forward(self, x):

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self._forward_mixer, x, use_reentrant=False)
        else:
            x = self._forward_mixer(x)
        x = shortcut + x
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self._forward_mlp, x, use_reentrant=False)
        else:
            x = x + self._forward_mlp(x)

        return x