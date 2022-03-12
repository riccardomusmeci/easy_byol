from argparse import ArgumentParser
from src.core.inference import inference

def parse_args():
    """Parses arguments from command line

    Returns:
        argparse.Namespace: arguments
    """
    parser = ArgumentParser("Self-Supervised-Learning inference script.")

    parser.add_argument(
        "--weights",
        type=str,
        help="path to .pth file"
    )

    parser.add_argument(
        "--params",
        type=str,
        help="path to hp.yml file"
    )

    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="directory to save the features extracted by the BYOL encoder into"
    )

    parser.add_argument(
        "--tsne",
        default=True,
        type=lambda x: True if x.lower() == "true" else False,
        help="enable TSNE computation for easier visualization (may take a while)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(args)