import torch
import torch_scatter
from torch import nn
from torch import LongTensor, Tensor
from torch_geometric.typing import OptTensor
import torch_geometric as pyg
from torch.autograd.functional import jacobian
from torch.func import hessian
from torch.autograd.functional import hessian as autograd_hessian
from typing import Optional
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
from loguru import logger
from math import ceil


class DiffModel(nn.Module):
    """A GNN is not differentiable in the edge_index argument so we can't call Jacobian on it,
    therefore we wrap it into another class which is differentiable in all its arguments,
    by storing the edge_index tensor"""

    def __init__(
        self,
        model: nn.Module,
        edge_index: LongTensor,
        pestat: Optional[list[Tensor]] = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        use_jacobian: bool = False,
    ):
        super(DiffModel, self).__init__()
        self.model = model
        self.edge_index = edge_index
        self.pestat = pestat
        self.edge_attr = edge_attr
        self.batch = batch
        self.use_jacobian = use_jacobian

    def forward(self, x):
        if self.use_jacobian:
            self.model.make_headless()
        result = self.model(
            x=x,
            edge_index=self.edge_index,
            pestat=self.pestat,
            edge_attr=self.edge_attr,
            batch=self.batch,
        )
        if self.use_jacobian:
            self.model.make_headless(undo=True)
        return result

    def jacobian(self, x):
        """
        Jacobian of the model with respect to x.
        If the model's output is [N, D] and input is [N, C] then the jacobian is a tensor of shape [N, D, N, C]
        """
        return jacobian(self.forward, x)

    def hessian(self, x):
        """
        Hessian of the model with respect to x.
        If the model's output is [B, K] and input is [N, D] then the hessian is a tensor of shape [B, K, N, D, N, D]
        """
        return hessian(self.forward)(x)

    def update_edge_index(self, new_edge_index):
        self.edge_index = new_edge_index

    def update_model(self, model):
        self.model = model


from torch_geometric.graphgym.config import cfg


class LRGBDiffModel(nn.Module):
    """
    A differentiable wrapper for GNN models that enables computing the Jacobian
    with respect to node features by encapsulating all necessary inputs from a DataBatch.
    Used for SOTA experiments on LRGB GraphGym tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        data_batch: pyg.data.Batch,
    ):
        """
        Initializes the LRGBDiffModel.

        Args:
            model (nn.Module): The GNN model to wrap.
            data_batch (pyg.data.Batch): The PyG DataBatch object containing graph data.
        """
        super(LRGBDiffModel, self).__init__()
        device = data_batch.x.device
        self.model = model.to(device)
        self.device = device
        self.split = data_batch.split
        self.chunk_size = (
            175  # hard coded value, will depend on the GPU memory available
        )
        try:
            self.num_classes = model.post_mp.mlp[-1].out_features
        except:
            pass

        if data_batch.num_nodes > self.chunk_size:
            # Otherwise throws a CUDA error
            self.chunk_jacobian = ceil(data_batch.num_nodes / self.chunk_size)
            logger.info(
                f"Chunking Jacobian computation into {self.chunk_jacobian} parts"
            )
        else:
            self.chunk_jacobian = None

        # Clone and store fixed attributes to prevent external mutations
        self.register_buffer("edge_index", data_batch.edge_index.clone().to(device))
        self.register_buffer(
            "edge_attr",
            (
                data_batch.edge_attr.clone().to(device)
                if data_batch.edge_attr is not None
                else None
            ),
        )
        self.register_buffer(
            "batch",
            (
                data_batch.batch.clone().to(device)
                if data_batch.batch is not None
                else None
            ),
        )
        self.register_buffer(
            "y",
            (data_batch.y.clone().to(device) if data_batch.y is not None else None),
        )
        self.register_buffer(
            "ptr",
            (data_batch.ptr.clone().to(device) if data_batch.ptr is not None else None),
        )
        if cfg.posenc_RWSE.enable:
            self.register_buffer(
                "pestat_RWSE",
                (
                    data_batch.pestat_RWSE.clone().to(device)
                    if data_batch.pestat_RWSE is not None
                    else None
                ),
            )
        if cfg.posenc_LapPE.enable:
            self.register_buffer(
                "EigVals",
                (
                    data_batch.EigVals.clone().to(device)
                    if data_batch.EigVals is not None
                    else None
                ),
            )
            self.register_buffer(
                "EigVecs",
                (
                    data_batch.EigVecs.clone().to(device)
                    if data_batch.EigVecs is not None
                    else None
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the wrapped model.

        Args:
            x (Tensor): Node feature tensor of shape [num_nodes, feature_dim].

        Returns:
            Tensor: Model output.
        """
        # Construct a new DataBatch with the input x and fixed attributes
        new_batch = pyg.data.Batch()
        new_batch.x = x
        new_batch.y = self.y
        new_batch.edge_index = self.edge_index
        new_batch.edge_attr = self.edge_attr
        new_batch.batch = self.batch
        new_batch.ptr = self.ptr
        new_batch.split = self.split
        if cfg.posenc_RWSE.enable:
            new_batch.pestat_RWSE = self.pestat_RWSE
        if cfg.posenc_LapPE.enable:
            new_batch.EigVals = self.EigVals
            new_batch.EigVecs = self.EigVecs

        return self.model(new_batch)[0]  # model returns pred, true

    def compute_jacobian(self, x: Tensor, chunk_idx=None) -> Tensor:
        """
        Computes the Jacobian of the model's output with respect to input features x.

        Args:
            x (Tensor): Input node feature tensor of shape [num_nodes, feature_dim].

        Returns:
            Tensor: Jacobian tensor.
                     If model output is [num_nodes, output_dim], Jacobian shape will be [num_nodes, output_dim, num_nodes, feature_dim].
        """
        # Ensure x requires gradient
        x_clone = x.requires_grad_(True).to(self.device)

        if self.chunk_jacobian:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided for chunked Jacobian")
            chunked_forward = lambda x0: self.forward(x0)[
                self.chunk_size * chunk_idx : self.chunk_size * (chunk_idx + 1)
            ]
            return jacobian(chunked_forward, x_clone)

        return jacobian(self.forward, x_clone)

    def compute_hessian(self, x: Tensor) -> Tensor:
        """
        Computes the Hessian of the model's output with respect to input features x.

        Args:
            x (Tensor): Input node feature tensor of shape [num_nodes, feature_dim].

        Returns:
            Tensor: Hessian tensor.
                     If model output is [num_nodes, output_dim], Hessian shape will be [num_nodes, output_dim, num_nodes, feature_dim, num_nodes, feature_dim].
        """
        # Ensure x requires gradient
        x = x.requires_grad_(True).to(self.device)

        abs_summed_hessian = 0
        for idx in tqdm(range(self.num_classes)):
            x_clone = x.clone().detach().requires_grad_(True).to(self.device)
            forward_wrapper = lambda x0: self.forward(x0).squeeze()[idx]
            hess = abs(autograd_hessian(forward_wrapper, x_clone))
            abs_summed_hessian += hess.detach()
            del hess, x_clone
            torch.cuda.empty_cache()

        return abs_summed_hessian


def compute_range(
    data: pyg.data.Data,
    range_mat: Tensor,
    norm: bool = False,
    dist: str = "res",
) -> Tensor:
    """
    Compute the range of the function, according to the specified distance function

    Args:
        data: pyg.data.Data
        range_mat (_type_): Jacobian or Hessian tensor of shape [N, N] where first dim is target, second is source.
            Already taken abs() and reduced from [N, dim_out, N, dim_in] (or [dim_out, N, dim_in, N, dim_in] for Hessian) to [N, N] by summing over dim_out and dim_in.
        norm: bool indicating whether to use the range or the normalized range
        dist: str with the name of the distance function to use

    Returns:
        range: tensor of shape [N], the range of the function around each node
    """
    device = range_mat.device
    if hasattr(data, "edge_index_dense"):
        edge_index_dist = data.edge_index_dense.to(device)
    else:
        edge_index_dist = dense_to_sparse(torch.ones([data.num_nodes, data.num_nodes]))[
            0
        ].to(device)
    if dist == "res":
        edge_attr_dist = data.edge_attr_res.to(device)
    elif dist == "spd":
        edge_attr_dist = data.edge_attr_spd.to(device)
    else:
        raise ValueError(f"Unknown distance function {dist}")

    dist = torch.zeros([data.num_nodes, data.num_nodes], device=device)
    dist[edge_index_dist[0], edge_index_dist[1]] = edge_attr_dist.squeeze()
    if norm:
        normalizer = range_mat @ torch.ones(range_mat.shape[0], device=range_mat.device)
        normalizer_inv = torch.pow(normalizer, -1).nan_to_num(0.0, posinf=0.0)
        range_mat_abs_norm = (
            torch.diag(normalizer_inv) @ range_mat
        )  # similar to random walk laplacian normalization
        rho = torch.einsum(
            "ab, ab -> a", range_mat_abs_norm, dist
        )  # [N] # see normalized range definition in paper
    else:
        rho = torch.einsum(
            "ab, ab -> a", range_mat, dist
        )  # [N] # see range definition in paper

    rho_mean = torch_scatter.scatter(
        rho, data.batch.to(device), reduce="mean"
    )  # [num_graphs] (1 for BS=1)
    rho_total = rho_mean.sum()

    return rho_total


class HeadlessGraphLevelModelWrapper(nn.Module):
    """
    A model wrapper for graph-level tasks that skips pooling and post-pooling
    so that per-node Jacobians over the whole model can be computed.
    """

    def __init__(self, original_model):
        super().__init__()
        self.encoder = original_model.model.encoder
        try:
            self.layers = original_model.model.gnn_layers
        except:
            self.layers = original_model.model.layers

    def forward(self, batch):
        # Run the encoder and GNN layers, skip the post_mp
        batch.x = batch.x.squeeze(0)  # Remove unnecessary batch dimension

        batch = self.encoder(batch)  # Assumes encoder takes *args/**kwargs
        batch = self.layers(batch)

        # Add batch dimension back so jacobian calc doens't auto-choose one
        x = batch.x.unsqueeze(0)  # -> [1, N, d]
        return x
