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

    cfg.longrange.split = "val"  # which split to evaluate for

    cfg.longrange.epoch = 0  # which epoch to evaluate for. 0-indexed

    # defines the probability of sampling input/output nodes/channels, the higher they are the more accurate the estimate.
    cfg.longrange.in_channel_prob = 1.0  # float between 0 and 1
    cfg.longrange.in_node_prob = 1.0
    cfg.longrange.out_channel_prob = 1.0  # out_channel_sample: float between 0 and 1
    cfg.longrange.out_node_prob = 1.0
    cfg.longrange.sampling_seed = 0  # seed for sampling channels/nodes
    # Node limits to avoid OOM when computing Jacobian, will depend on user's compute resources
    # To compute true Jacobian, set min_nodes=max_nodes=0
    cfg.longrange.min_nodes = 16  # min(/max) number of nodes to sample
    cfg.longrange.max_nodes = 256  # 0 defaults to inf

    cfg.dataset.virtual_node = False  # whether to add a VN or not
