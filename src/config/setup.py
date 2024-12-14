# https://docs.python.org/3/library/argparse.html

import argparse


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Application for recognize mushroom image from photo")

    # Download and unzip dataset
    parser.add_argument(
        "--download",
        "-d",
        action="store_true",
        help="Download dataset.",
    )

    # Train model
    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        help="Train model",
    )

    # Predict model
    parser.add_argument("--predict", "-p", action="store_true", help="Predict model")

    return parser.parse_args()
