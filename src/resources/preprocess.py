import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.config.path import Path
from src.graphs import Graphs
from src.resources.manager import Manager


class Preprocess:
    def __init__(self, mushrooms_path: str) -> None:
        self.mushrooms_path = mushrooms_path
        self.data = pd.DataFrame()

    def get_image(self) -> None:
        """
        Collects image file paths and their corresponding class names from the dataset directory
        and stores them in a DataFrame.

        Returns:
        None: Updates the `self.data` attribute with a DataFrame containing:
            - 'filepath': The full file path to the image.
            - 'name': The class name (folder name).
        """
        image_list = []

        for filepath in Manager.get_images_from_folder(self.mushrooms_path):
            class_name = os.path.basename(os.path.dirname(filepath))
            image_list.append((filepath, class_name))

        self.data = pd.DataFrame(image_list, columns=["filepath", "name"])

    def get_classes(self) -> np.ndarray:
        """
        Retrieves a list of unique class names from the dataset.

        Returns:
            numpy.ndarray: A list of unique class names extracted from the 'name' column of the DataFrame.
        """
        return self.data["name"].unique()

    def balance_classes(self) -> None:
        """
        Balances the dataset by undersampling larger classes to match the size of the smallest class.

        Returns:
            None: This method updates the `self.data` attribute in place.
        """
        min_class_size = self.data["name"].value_counts().min()
        balanced_data = pd.DataFrame(columns=self.data.columns)

        for mushroom_class in self.get_classes():
            class_data = self.data[self.data["name"] == mushroom_class]
            if len(class_data) > min_class_size:
                # Undersampling
                class_data = resample(class_data, replace=False, n_samples=min_class_size, random_state=42)

            balanced_data = pd.concat([balanced_data, class_data], ignore_index=True)

        self.data = balanced_data

    def create_data_for_learning(self) -> tuple:
        """
        Splits the dataset into training, validation, and test datasets for machine learning.

        Returns:
            tuple: A tuple containing three pandas DataFrames:
                - train_data: DataFrame with training data.
                - val_data: DataFrame with validation data.
                - test_data: DataFrame with test data.
        """
        mushroom_classes = self.get_classes()

        train_data = pd.DataFrame(columns=["filepath", "name"])
        val_data = pd.DataFrame(columns=["filepath", "name"])
        test_data = pd.DataFrame(columns=["filepath", "name"])

        # Creating training and test data
        full_train_data = pd.DataFrame(columns=["filepath", "name"])
        for mushroom in mushroom_classes:
            temp = self.data[self.data["name"] == mushroom].copy()
            train, test = train_test_split(temp, test_size=0.1, train_size=0.9)

            train_ls = train[["name", "filepath"]]
            test_ls = test[["name", "filepath"]]

            full_train_data = pd.concat([full_train_data, train_ls], ignore_index=True, sort=False)
            test_data = pd.concat([test_data, test_ls], ignore_index=True, sort=False)

        # Creating training, validation, and test data
        train_data = pd.DataFrame(columns=["filepath", "name"])
        val_data = pd.DataFrame(columns=["filepath", "name"])
        for mushroom in mushroom_classes:
            temp = full_train_data[full_train_data["name"] == mushroom].copy()
            train, test = train_test_split(temp, test_size=0.09)

            train_ls = train[["name", "filepath"]]
            test_ls = test[["name", "filepath"]]

            train_data = pd.concat([train_data, train_ls], ignore_index=True, sort=False)
            val_data = pd.concat([val_data, test_ls], ignore_index=True, sort=False)

        return (train_data, val_data, test_data)

    def execute(self) -> tuple:
        """
        Preporcess data.

        Returns:
            tuple: A tuple containing three pandas DataFrames:
                - train_data: DataFrame with training data.
                - val_data: DataFrame with validation data.
                - test_data: DataFrame with test data.
        """
        # Create an image array
        self.get_image()
        # Generate a report
        Graphs.images_count(self.data, Path.graphs_images_count, "Mushroom class distribution")

        # Balance the number of images
        self.balance_classes()
        # Generate a report after balancing
        Graphs.images_count(self.data, Path.graphs_images_balances_count, "Mushroom class distribution balanced")

        return self.create_data_for_learning()
