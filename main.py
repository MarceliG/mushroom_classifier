from src import Predict, Train
from src.config import Path, parser
from src.logs import logger
from src.resources import Manager


def main() -> None:
    logger.info("Start application")
    args = parser()

    if args.download:
        Manager.download_dataset(
            author="maysee",
            dataset_name="mushrooms-classification-common-genuss-images",
            save_loaction=Path.datasets,
            extract_folder="Mushrooms",
        )

    if args.train:
        train = Train()
        train.execute(Path.mushrooms, Path.models_filename, Path.models_json)

    if args.predict:
        predict = Predict()
        predict.execute(image_folder=Path.predict_path)

    logger.info("Finish application")


if __name__ == "__main__":
    main()
