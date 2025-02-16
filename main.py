from torch import cuda
from helpers.parse_arguments import parse_arguments
from experiments import Experiment
from loguru import logger

if __name__ == "__main__":
    args = parse_arguments()
    if args.gpu is not None:
        if cuda.is_available():
            cuda.set_device(args.gpu)
        else:
            logger.warning("CUDA is not available, defaulting to CPU.")
    Experiment(args=args).run()
