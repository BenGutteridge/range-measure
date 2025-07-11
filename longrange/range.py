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
from helpers.gpu_management.gpu_utils import GPUTracker
import time


class DiffModel(nn.Module):
    """
    A differentiable wrapper for GNN models that enables computing the Jacobian
    with respect to node features by encapsulating all necessary inputs from a DataBatch.
    Used for SOTA experiments on LRGB GraphGym tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        data_batch: pyg.data.Batch,
        output_dim: int,
        use_jacobian: bool = False,
        lrgb_mode: bool = False,
    ):
        """
        Initializes the LRGBDiffModel.

        Args:
            model (nn.Module): The GNN model to wrap.
            data_batch (pyg.data.Batch): The PyG DataBatch object containing graph data.
            output_dim (int): The output dimension of the model.
            pestat (Optional[list[Tensor]], optional): Positional encoding statistics. Defaults to None.
            use_jacobian (bool, optional): Flag to indicate if Jacobian should be used. Defaults to False.
        """
        super(DiffModel, self).__init__()
        
        self.model = model
        self.use_jacobian = use_jacobian
        self.output_dim = output_dim
        self.device = data_batch.x.device
        if self.device != torch.device("cpu"):
            self.gpu_tracker = GPUTracker(self.device)
        self.lrgb_mode = lrgb_mode

        for attr in [
            "edge_index",
            "edge_attr",
            "batch",
            "y",
            "ptr",
            "pestat_RWSE",
            "EigVals",
            "EigVecs",
        ]:
            if hasattr(data_batch, attr):
                self.register_buffer(
                    attr,
                    getattr(data_batch, attr).clone().to(self.device)
                    if getattr(data_batch, attr) is not None
                    else None,
                )
        # pestat taken as an input to models.MPNN.MPNN() forward pass
        if hasattr(self, "pestat_RWSE"):
            self.pestat = self.pestat_RWSE
        elif hasattr(self, "EigVals") and hasattr(self, "EigVecs"):
            self.pestat = [self.EigVals, self.EigVecs]
        else:
            self.pestat = None
        

    def forward(self, x: Tensor):
        if self.lrgb_mode:
            return self.forward_lrgb(x)
        
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

    def forward_lrgb(self, x: Tensor) -> Tensor:
        """
        Forward pass of the wrapped model.

        Args:
            x (Tensor): Node feature tensor of shape [num_nodes, feature_dim].

        Returns:
            Tensor: Model output.
        """
        new_batch = pyg.data.Batch()
        new_batch.x = x
        for attr in [
            "y",
            "edge_index",
            "edge_attr",
            "batch",
            "ptr",
            "split",
            "pestat_RWSE",
            "EigVals",
            "EigVecs",
        ]:
            if hasattr(self, attr):
                setattr(new_batch, attr, getattr(self, attr))

        return self.model(new_batch)[0]  # model returns (pred, true); we only want pred
    

    def compute_jacobian(self, x: Tensor) -> Tensor:
        """
        Computes the Jacobian of the model's output with respect to input features x.

        Args:
            x (Tensor): Input node feature tensor of shape [num_nodes, feature_dim].

        Returns:
            Tensor: Jacobian tensor.
                     If model output is [num_nodes, output_dim], Jacobian shape will be [num_nodes, output_dim, num_nodes, feature_dim].
        """
        x_clone = x.requires_grad_(True).to(self.device)
        return jacobian(self.forward, x_clone)
    

    def sample_jacobian(
        self,
        x: Tensor,
        in_node_mask: Tensor,
        in_dim_mask: Tensor,
        out_node_mask: Tensor,
        out_dim_mask: Tensor,
    ) -> Tensor:
        """
        Grid samples the Jacobian of the model's output with respect to input features x.

        Args:
            x (Tensor): Input node feature tensor of shape [num_nodes, feature_dim].
            num_in_channels (int): number of input channels to sample.
            num_out_channels (int): number of output channels to sample.
            num_in_nodes (int): number of input nodes to sample.
            num_out_nodes (int): number of output nodes to sample.
            sampling_seed (int): random seed for sampling.

        Returns:
            Tensor: Jacobian tensor.
                     Jacobian shape will be [num_nodes, out_channels, num_nodes, in_channels]].
        """
        # Ensure x requires gradient
        x_clone = x.requires_grad_(True).to(self.device)

        # Insert only sampled values of x into the cloned overall x
        def forward_wrapper(x0):
            x_updated = x_clone.detach()
            x_updated[in_node_mask[:, None], in_dim_mask] = x0
            return self.forward(x_updated)[out_node_mask[:, None], out_dim_mask]

        x_in = x_clone[in_node_mask, :][:, in_dim_mask]
        
        logger.info(
            f"Sampling {out_node_mask.shape[0]}x{out_dim_mask.shape[0]}x{in_node_mask.shape[0]}x{in_dim_mask.shape[0]} Jacobian..."
        )
        t0 = time.time()
        sampled_jac = jacobian(forward_wrapper, x_in)  # .detach().cpu()
        logger.info(f"Obtained {sampled_jac.shape} Jacobian in {time.time()-t0:.2f}s")

        abs_summed_sampled_jac = sampled_jac.detach().cpu().abs().sum(dim=1).sum(dim=2)

        return abs_summed_sampled_jac
    
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



def compute_range(
    data: pyg.data.Data,
    range_mat: Tensor,
    norm: bool,
    dist: str,
    out_dim: int,
    virtual_node: bool = False,
    **sample_kwargs,
) -> Tensor:
    """
    Compute the range of the function, according to the specified distance function

    Args:
        data: pyg.data.Data
        range_mat (_type_): Jacobian or Hessian tensor of shape [N, N] where first dim is target, second is source.
            Already taken abs() and reduced from [N, dim_out, N, dim_in] (or [dim_out, N, dim_in, N, dim_in] for Hessian) to [N, N] by summing over dim_out and dim_in.
        norm: bool indicating whether to use the range or the normalized range
        dist: str with the name of the distance function to use
        out_dim: int, the total number of output channels
        sample_kwargs: additional arguments to pass to the range computation: num_in_nodes, num_out_nodes, num_in_channels, num_out_channels (all ints)

    Returns:
        range: tensor of shape [N], the range of the function around each node
    """
    device = range_mat.device
    num_nodes = range_mat.shape[0]
    if virtual_node:
        if num_nodes == data.num_nodes - 1:
            # virtual node is implemented as a separate node in the data
            # remove the last node
            remove_last = True
        elif num_nodes == data.num_nodes:
            # virtual node is implemented within the model
            remove_last = False
        else:
            raise ValueError(f"Node mismatch: range mat is {num_nodes} vs {data.num_nodes=}")
    num_in_nodes = sample_kwargs.get("num_in_nodes", num_nodes)
    num_out_nodes = sample_kwargs.get("num_out_nodes", num_nodes)
    num_in_channels = sample_kwargs.get("num_in_channels", data.x.shape[1])
    num_out_channels = sample_kwargs.get("num_out_channels", out_dim)

    if hasattr(data, "edge_index_dense"):
        edge_index_dist = data.edge_index_dense.to(device)
    else:
        edge_index_dist = dense_to_sparse(torch.ones([num_nodes, num_nodes]))[0].to(
            device
        )
    if dist == "res":
        edge_attr_dist = data.edge_attr_res.to(device)
    elif dist == "spd":
        edge_attr_dist = data.edge_attr_spd.to(device)
    else:
        raise ValueError(f"Unknown distance function {dist}")

    dist = torch.zeros([num_nodes, num_nodes], device=device)
    dist[edge_index_dist[0], edge_index_dist[1]] = edge_attr_dist.squeeze()
    if norm:
        # Divide each Jacobian element by the row sum
        normalizer = range_mat @ torch.ones(
            range_mat.shape[0], device=range_mat.device
        )  # row sum
        normalizer_inv = torch.pow(normalizer, -1).nan_to_num(0.0, posinf=0.0)
        range_mat_abs_norm = (
            torch.diag(normalizer_inv) @ range_mat
        )  # similar to random walk laplacian normalization
        rho = torch.einsum(  # elementwise product followed by column sum
            "ab, ab -> a", range_mat_abs_norm, dist
        )  # [N] # see normalized range definition in paper
    else:
        rho = torch.einsum(
            "ab, ab -> a", range_mat, dist
        )  # [N] # see range definition in paper
        num_in_nodes = num_nodes if num_in_nodes is None else num_in_nodes
        total_in_channels = data.x.shape[1]
        rho *= (total_in_channels * out_dim * num_nodes) / (
            num_in_channels * num_out_channels * num_in_nodes
        )  # scale the estimate to account for unsampled nodes/channels

    if virtual_node:
        assert (
            len(data.batch.unique()) == 1
        ), "Virtual node only supported for range computation with batch size 1"
        if remove_last:
            batch = data.batch.to(device)[:-1]
        else:
            batch = data.batch.to(device)
    else:
        batch = data.batch.to(device)
    rho_sum = torch_scatter.scatter(
        rho, batch, reduce="sum"
    )  # [num_graphs] (1 for BS=1) # sum over node range in each graph
    rho_mean = rho_sum / num_out_nodes
    return rho_mean


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
        # skip post_mp

    def forward(self, batch):
        # Run the encoder and GNN layers, skip the post_mp
        batch.x = batch.x.squeeze(0)  # Remove unnecessary batch dimension

        batch = self.encoder(batch)  # Assumes encoder takes *args/**kwargs
        batch = self.layers(batch)

        # Add batch dimension back so jacobian calc doens't auto-choose one
        x = batch.x.unsqueeze(0)  # -> [1, N, d]
        return x