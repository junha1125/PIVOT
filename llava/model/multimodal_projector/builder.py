import torch
import torch.nn as nn
import re

from .pooler_projector import PoolerProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        source_load = getattr(config, "mm_projector_source_load_layers", 0)
        source_hidden = getattr(config, "mm_projector_source_hidden_size", None)
        need_bridge = (source_load > 0
                       and source_hidden is not None
                       and source_hidden != config.hidden_size)

        if need_bridge:
            assert source_load < mlp_depth, (
                f"Cannot load all {mlp_depth} projector layers from source when "
                f"hidden_size differs (source={source_hidden}, current={config.hidden_size}). "
                f"Set mm_projector_source_load_layers < {mlp_depth}."
            )
            modules = []
            for i in range(mlp_depth):
                in_d = config.mm_hidden_size if i == 0 else (source_hidden if i <= source_load else config.hidden_size)
                out_d = source_hidden if i < source_load else config.hidden_size
                modules.append(nn.Linear(in_d, out_d))
                if i < mlp_depth - 1:
                    modules.append(nn.GELU())
        else:
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))

        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
