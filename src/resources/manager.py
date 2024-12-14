import json
import os
import zipfile
from typing import Any

import kaggle
from tensorflow.keras.utils import Sequence

from src.config import Path
from src.logs import logger


class Manager:
    @staticmethod
    def download_dataset(author: str, dataset_name: str, save_loaction: str, extract_folder: str = "") -> None:
        """
        Downloads and extracts a dataset from Kaggle.

        Args:
            author (str): The author or uploader of the dataset on Kaggle.
            dataset_name (str): The name of the dataset on Kaggle.
            save_location (str): The directory where the dataset will be saved.
            extract_folder (str, optional): The folder to extract the dataset into. Defaults to an empty string.

        Returns:
            None
        """
        # Need Kaggle account to dowload it
        api = kaggle.KaggleApi()
        api.authenticate()

        # Download
        logger.info("Start downloading dataset")
        full_dataset_name = f"{author}/{dataset_name}"
        api.dataset_download_files(full_dataset_name, path=save_loaction)

        # Unzip
        logger.info(f"Unzip folder {extract_folder}")
        Manager.unzip_dataset(dataset_name, save_loaction, extract_folder)

    @staticmethod
    def unzip_dataset(dataset_name: str, save_loaction: str, extract_folder: str = "") -> None:
        """
        Extracts a specified dataset from a ZIP file.

        Args:
            dataset_name (str): The name of the dataset (without the `.zip` extension).
            save_location (str): The directory where the ZIP file is located and where files will be extracted.
            extract_folder (str, optional): Specific folder within the ZIP to extract. Extracts all if empty.

        Returns:
            None
        """
        zip_path = os.path.join(save_loaction, f"{dataset_name}.zip")

        # Unzip the dataset
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            if extract_folder:
                for file in zip_ref.namelist():
                    if file.startswith(extract_folder):
                        zip_ref.extract(file, save_loaction)
            else:
                zip_ref.extractall(save_loaction)

        # Remove zip file
        os.remove(zip_path)

    @staticmethod
    def save_classes_as_json(json_path: str, train_ds: Sequence) -> None:
        classes = list(train_ds.class_indices.keys())

        with open(json_path, "w") as f:
            json.dump(classes, f)

        logger.info(f"Classes saved to {json_path}")

    @staticmethod
    def load_json_classes() -> Any:
        with open(Path.models_json) as f:
            return json.load(f)
