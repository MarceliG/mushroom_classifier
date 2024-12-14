import glob
import os

import numpy as np
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

from src.config import Path
from src.logs import logger
from src.resources import Manager
from src.train import Train


class Predict:
    def __init__(self) -> None:
        self.model_path = Path.models_filename
        self.model = load_model(self.model_path)
        self.classes = Manager.load_json_classes()

    def get_images_from_folder(self, folder_path: str) -> list:
        """
        Retrieves all image file paths from a folder.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            list: List of file paths for all images in the folder.
        """
        return glob.glob(os.path.join(folder_path, "**/*.jpg"), recursive=True)

    def execute(self, image_folder: str) -> None:
        """
        Executes the prediction pipeline on a folder of images.

        Args:
            image_folder (str): Path to the folder containing images.

        Returns:
            None
        """
        # Get all image paths from the folder
        image_paths = self.get_images_from_folder(image_folder)

        if not image_paths:
            logger.info("No images found in the specified folder.")
            return

        logger.info(f"Found {len(image_paths)} image(s) in the folder.")

        for image_path in image_paths:
            # Load image and preprocess
            img = self.preprocess_image(image_path)

            # Make prediction
            pred = self.model.predict(img)
            result: dict[str, float] = dict(zip(self.classes, pred[0]))
            max_class = max(result, key=lambda k: result[k])

            # Log prediction
            logger.info(
                f"For image: {os.path.basename(image_path)} => Predicted class: {max_class}, Score: {result[max_class]:.4f}"
            )

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Loads and preprocesses an image for model prediction.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image ready for prediction.
        """
        img = load_img(image_path, target_size=(Train.img_size, Train.img_size))
        array_img = np.array(img, dtype=np.float32)

        return preprocess_input(np.expand_dims(array_img, axis=0))
