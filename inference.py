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
        "--weights",
        default="checkpoints/byol/byol_2021-12-19-11-59-58/byol_resnet18_epoch_0_loss_0.0323.pth",
        type=str,
        help="path to .pth file"
    )

    parser.add_argument(
        "--gradcam",
        type=lambda x: True if x.strip().lower() == "true" else False,
        default=True,
        metavar="N",
        help="whether to apply gradcam."
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if args.model == "byol":
        from src.core.byol_inference import inference
        inference(args)

    else:
        print(f"No implementation for {args.model}")