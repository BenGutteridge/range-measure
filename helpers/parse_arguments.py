from argparse import ArgumentParser

from helpers.dataset_classes.dataset import Dataset
from helpers.model import ModelType
from helpers.classes import Pool
from helpers.encoders import PosEncoder


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="dataset",
        default=Dataset.roman_empire,
        type=Dataset.from_string,
        choices=list(Dataset),
        required=False,
    )
    parser.add_argument(
        "--pool",
        dest="pool",
        default=Pool.NONE,
        type=Pool.from_string,
        choices=list(Pool),
        required=False,
    )

    # gumbel
    parser.add_argument(
        "--learn_temp",
        dest="learn_temp",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--temp_model_type",
        dest="temp_model_type",
        default=ModelType.LIN,
        type=ModelType.from_string,
        choices=list(ModelType),
        required=False,
    )
    parser.add_argument("--tau0", dest="tau0", default=0.5, type=float, required=False)
    parser.add_argument("--temp", dest="temp", default=0.01, type=float, required=False)

    # optimization
    parser.add_argument(
        "--max_epochs", dest="max_epochs", default=3000, type=int, required=False
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", default=32, type=int, required=False
    )
    parser.add_argument("--lr", dest="lr", default=1e-3, type=float, required=False)
    parser.add_argument(
        "--dropout", dest="dropout", default=0.0, type=float, required=False
    )

    # gnn cls parameters
    parser.add_argument(
        "--model_type",
        dest="model_type",
        default=ModelType.GCN,
        type=ModelType.from_string,
        choices=list(ModelType),
        required=False,
    )
    parser.add_argument(
        "--num_layers", dest="num_layers", default=3, type=int, required=False
    )
    parser.add_argument(
        "--hidden_dim", dest="hidden_dim", default=64, type=int, required=False
    )
    parser.add_argument(
        "--skip", dest="skip", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--batch_norm",
        dest="batch_norm",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--layer_norm",
        dest="layer_norm",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--dec_num_layers", dest="dec_num_layers", default=1, type=int, required=False
    )
    parser.add_argument(
        "--pos_enc",
        dest="pos_enc",
        default=PosEncoder.NONE,
        type=PosEncoder.from_string,
        choices=list(PosEncoder),
        required=False,
    )

    # reproduce
    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)
    parser.add_argument("--gpu", dest="gpu", default=0, type=int, required=False)

    # dataset dependant parameters
    parser.add_argument("--fold", dest="fold", default=None, type=int, required=False)

    # optimizer and scheduler
    parser.add_argument(
        "--weight_decay", dest="weight_decay", default=0, type=float, required=False
    )
    ## for steplr scheduler only
    parser.add_argument(
        "--step_size", dest="step_size", default=None, type=int, required=False
    )
    parser.add_argument(
        "--gamma", dest="gamma", default=None, type=float, required=False
    )
    ## for cosine with warmup scheduler only
    parser.add_argument(
        "--num_warmup_epochs",
        dest="num_warmup_epochs",
        default=None,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--evaluate_only",
        dest="evaluate_only",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        default=None,
        type=str,
        required=False,
    )
    # --------------------------------------------------------------------------------------
    # ------------------- For LR experiments -----------------------------------------------
    # --------------------------------------------------------------------------------------
    # For synthetic experiments
    parser.add_argument(
        "--distance_fn",
        dest="distance_fn",
        default="adjacency_matrix",
        type=str,
        required=False,
    )  # distance function to use for LR synthetic task

    parser.add_argument(
        "--interaction_fn",
        dest="interaction_fn",
        default="L2_norm",
        type=str,
        required=False,
    )  # interaction function to use for LR synthetic task

    parser.add_argument(
        "--num_graphs", dest="num_graphs", default=500, type=int, required=False
    )  # number of graphs to use in dataset for synthetic experiments

    # --------------------------------------------------------------------------------------
    # For tracking range measures experiments
    parser.add_argument(
        "--track_range",
        dest="track_range",
        default=False,
        action="store_true",
        required=False,
    )  # whether or not to track the Jacobian of the model during training

    parser.add_argument(
        "--range_splits",
        dest="range_splits",
        default=["train", "val", "test"],
        nargs="+",
        choices=["train", "val", "test"],
        help="Split(s) to track the range measure on. Choose one or more: train, val, test. Defaults to all three.",
    ) # example usage: --range_splits train val -> track range on train and val splits

    parser.add_argument(
        "--track_epoch_every", 
        dest="track_epoch_every", 
        default=1, 
        type=int, 
        required=False,
    ) # Tracks every nth epoch

    # Option to add a list of integers to track specific epochs
    parser.add_argument(
        "--track_epochs",
        nargs="+",
        required=False,
        default=[],
        type=int,
    )  # example useage: --track_epochs 1 2 3 4 5

    parser.add_argument(
        "--wandb",
        dest="wandb",
        default=False,
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--wandb_project",
        dest="wandb_project",
        default="Range-all",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--subset",
        dest="subset",
        default=None,
        type=int,
        required=False,
        help="Limit the dataset processing to the first N graphs (integer). Defaults to None for full dataset.",
    )

    parser.add_argument(
        "--use_jacobian",
        dest="use_jacobian",
        default=False,
        action="store_true",
        required=False,
        help="If true then it will use the final layer's jacobian to compute the range, if false it uses the hessian of the output.",
    )

    parser.add_argument(
        "--virtual_node",
        dest="virtual_node",
        default=False,
        action="store_true",
        required=False,
        help="Adds a VN",
    )

    # Params for sampling Jacobian for range exps for large graphs for heterophilic tasks
    parser.add_argument(
        "--in_channel_sample_prob",
        dest="in_channel_sample_prob",
        default=1.0,
        type=float,
        required=False,
        help="Probability of sampling an in-channel.",
    )
    parser.add_argument(
        "--out_channel_sample_prob",
        dest="out_channel_sample_prob",
        default=1.0,
        type=float,
        required=False,
        help="Probability of sampling an out-channel.",
    )
    parser.add_argument(
        "--in_node_sample_prob",
        dest="in_node_sample_prob",
        default=1.0,
        type=float,
        required=False,
        help="Probability of sampling an in-node.",
    )
    parser.add_argument(
        "--out_node_sample_prob",
        dest="out_node_sample_prob",
        default=1.0,
        type=float,
        required=False,
        help="Probability of sampling an out-node.",
    )

    return parser.parse_args()
