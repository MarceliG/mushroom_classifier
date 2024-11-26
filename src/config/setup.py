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
        "--preprocessing",
        "-p",
        action="store_true",
        help="Perform data preprocessing.",
    )

    return parser.parse_args()