from argparse import ArgumentParser

def parse_args():
    """Parses arguments from command line

    Returns:
        argparse.Namespace: arguments
    """
    parser = ArgumentParser("Self-Supervised-Learning training script.")
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model to train"
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
        default= "checkpoints",
        metavar="N",
        help="Path to the local directory where checkpoints are saved during training."
    )

    parser.add_argument(
        "--seed",
        type=int, 
        default=42, 
        metavar="N", 
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

    if args.model == "byol":
        from src.core.byol_train import train
        train(args)
    else:
        print(f"No implementation for {args.model}")
