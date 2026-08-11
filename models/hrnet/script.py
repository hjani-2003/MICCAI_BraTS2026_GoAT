import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels, requested_groups):
    groups = min(requested_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _match_spatial_size(x, reference):
    if x.shape[2:] == reference.shape[2:]:
        return x
    return F.interpolate(x, size=reference.shape[2:], mode="trilinear", align_corners=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, padding=dilation, dilation=dilation, bias=False,
        )
        self.gn1 = nn.GroupNorm(_valid_group_count(out_channels, groups), out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, padding=dilation, dilation=dilation, bias=False,
        )
        self.gn2 = nn.GroupNorm(_valid_group_count(out_channels, groups), out_channels)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = self.gn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class HRStage(nn.Module):

    def __init__(self, h_ch, m_ch, blocks_per_stage=2):
        super().__init__()
        self.h_blocks = nn.Sequential(*[ResidualBlock(h_ch, h_ch) for _ in range(blocks_per_stage)])
        self.m_blocks = nn.Sequential(*[ResidualBlock(m_ch, m_ch) for _ in range(blocks_per_stage)])

        # M -> H: channel projection (spatial upsample happens in forward)
        self.m_to_h = nn.Sequential(
            nn.Conv3d(m_ch, h_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_group_count(h_ch, 8), h_ch),
        )
        # H -> M: stride-2 conv + channel expansion
        self.h_to_m = nn.Sequential(
            nn.Conv3d(h_ch, m_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_valid_group_count(m_ch, 8), m_ch),
        )

    def forward(self, h, m):
        h = self.h_blocks(h)
        m = self.m_blocks(m)

        # Cross-branch contributions computed from pre-fusion h and m
        m_contrib = self.m_to_h(
            F.interpolate(m, size=h.shape[2:], mode="trilinear", align_corners=False)
        )
        h_contrib = _match_spatial_size(self.h_to_m(h), m)

        h_out = F.relu(h + m_contrib)
        m_out = F.relu(m + h_contrib)

        return h_out, m_out


class SmallTumorHRSeg(nn.Module):

    def __init__(
        self,
        in_channels=4,
        num_classes=3,
        h_ch=32,
        m_ch=64,
        num_stages=3,
        blocks_per_stage=2,
    ):
        super().__init__()

        # Stem: first feature extraction at full resolution
        self.stem = ResidualBlock(in_channels, h_ch)

        # Spawn M branch from stem output via strided conv
        self.init_m = nn.Sequential(
            nn.Conv3d(h_ch, m_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_valid_group_count(m_ch, 8), m_ch),
            nn.ReLU(inplace=True),
        )

        # HR stages with bidirectional fusion
        self.stages = nn.ModuleList([
            HRStage(h_ch, m_ch, blocks_per_stage) for _ in range(num_stages)
        ])

        # Head: bring M up to full resolution, cat with H, predict
        self.head_proj_m = nn.Sequential(
            nn.Conv3d(m_ch, h_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_group_count(h_ch, 8), h_ch),
            nn.ReLU(inplace=True),
        )
        self.head_fuse = ResidualBlock(h_ch * 2, h_ch)
        self.seg_head = nn.Conv3d(h_ch, num_classes, kernel_size=1)

    def forward(self, x):
        h = self.stem(x)       # [B, h_ch, D,   H,   W  ]
        m = self.init_m(h)     # [B, m_ch, D/2, H/2, W/2]

        for stage in self.stages:
            h, m = stage(h, m)

        m_up = self.head_proj_m(
            F.interpolate(m, size=h.shape[2:], mode="trilinear", align_corners=False)
        )

        fused = self.head_fuse(torch.cat([h, m_up], dim=1))
        return self.seg_head(fused)
