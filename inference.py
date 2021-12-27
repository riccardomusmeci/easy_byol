from argparse import ArgumentParser

def parse_args():
    """Parses arguments from command line

    Returns:
        argparse.Namespace: arguments
    """
    parser = ArgumentParser("Self-Supervised-Learning inference script.")
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
        "--weights",
        type=str,
        help="path to .pth file"
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

    if args.model == "byol":
        from src.core.byol_inference import inference
        inference(args)

    else:
        print(f"No implementation for {args.model}")