from argparse import ArgumentParser
from src.core.train_classifier import train

def parse_args():
    """Parses arguments from command line

    Returns:
        argparse.Namespace: arguments
    """
    parser = ArgumentParser()
    
    parser.add_argument(
        "--ssl-weights",
        type=str,
        default="checkpoints/byol/byol_2021-12-19-11-59-58/byol_resnet18_epoch_0_loss_0.0323.pth",
        help="path to byol trained weights for the encoder."
    )

    parser.add_argument(
        "--hp-dir",
        type=str,
        default="hp",
        help="Dir containing hp.yml for each model. Default to hp."
    )

    parser.add_argument(
        "--val-period",
        type=int,
        default=1,
        help="After how many epochs performing validation step. Defaults to 1."
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Path to the local directory where checkpoints are saved during training."
    )

    parser.add_argument(
        "--seed",
        type=int, 
        default=42, 
        help="Random seed. Default: 42"
    )

    parser.add_argument(
        "--save-disk",
        default=True,
        type=lambda x: True if x.lower() == "true" else False,
        help="save disk mode when saving pth checkpoints model"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
