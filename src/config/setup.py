# https://docs.python.org/3/library/argparse.html

import argparse

def parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Application for recognize image from photo"
    )

    parser.add_argument(
        "--download",
        "-d",
        action="store_true",
        help="Download dataset.",
    )

    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        help="Train data",
    )

    return parser.parse_args()