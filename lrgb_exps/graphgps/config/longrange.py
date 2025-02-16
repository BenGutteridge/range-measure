from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("longrange")
def set_cfg_example(cfg):
    r"""
    Settings for long-range experiments
    """

    cfg.longrange = CN()

    cfg.longrange.track_range_measure = False

    # Use a subset of the LRGB dataset (same number of graphs for all splits)
    cfg.longrange.subset = (
        -1
    )  # default, use normal dataset. subset >= minimum subset of 64 is needed

    cfg.longrange.split = "train"  # which split to evaluate for

    cfg.longrange.epoch = 0  # which epoch to evaluate for. 0-indexed
