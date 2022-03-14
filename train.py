from argparse import ArgumentParser
from src.core.byol.train import train as train_byol
from src.core.classifier.train import train as train_classifier

def parse_args():
    """Parses arguments from command line

    Returns:
        argparse.Namespace: arguments
    """
    parser = ArgumentParser()
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="byol or classifier training"
    )
    
    parser.add_argument(
        "--ssl-dir",
        type=str,
        default="checkpoints/byol/byol_2022-03-13-12-51-30",
        help="path to byol dir with training data (weights/*.pth and config.yml) for the encoder. Only if --model is set to classifier."
    )
    
    parser.add_argument(
        "--ssl-pth",
        type=str,
        default="byol_resnet18_epoch_0_loss_0.0323.pth",
        help="encoder pth file in ssl weights dir. Only if --model is set to classifier."
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Dir containing config.yml for each model. Default to config."
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
        help="save disk space when saving pth checkpoints model by having always the best model."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.model == "byol":
        train_byol(args)
    elif args.model == "classifier":
        train_classifier(args)
