import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss("weighted_cross_entropy")
def weighted_cross_entropy_wrapper(pred, true):
    "Wrapper for weighted cross-entropy that returns None if it is not the chosen loss_func in the config, to be compatible with graphgym."
    if cfg.model.loss_fun == "weighted_cross_entropy":
        return weighted_cross_entropy(pred, true)


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes."""
    # Calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # if weights all zero, set to uniform, otherwise loss will be nan
    if weight.sum() == 0:
        weight = None
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight), pred
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(
            pred, true.float(), weight=weight[true]
        )
        return loss, torch.sigmoid(pred)


def weighted_cross_entropy_loss():
    """Wrapper for weighted cross-entropy that returns only the loss."""
    return lambda pred, true: weighted_cross_entropy(pred, true)[0]
