from argparse import ArgumentParser
from src.core.byol.inference import inference as inference_byol
from src.core.classifier.inference import inference as inference_classifier


def parse_args():
    """Parses arguments from command line

    Returns:
        argparse.Namespace: arguments
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="byol/classifier"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        help="path to .pth file"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="path to config.yml file"
    )

    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="directory to save the output of the model"
    )

    parser.add_argument(
        "--tsne",
        default=True,
        type=lambda x: True if x.lower() == "true" else False,
        help="enable TSNE computation for easier visualization (may take a while). Only for BYOL model."
    )
    
    parser.add_argument(
        "--gradcam",
        default=True,
        type=lambda x: True if x.lower() == "true" else False,
        help="enable gradcam images saving. Only for classifier model."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.model == "byol":
        inference_byol(args)
    
    elif args.model == "classifier":
        inference_classifier(args)