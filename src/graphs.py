import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import History

from src.config.path import Path


class Graphs:
    @staticmethod
    def images_count(data: pd.DataFrame, save_paths: str, graph_title: str) -> None:
        """
        Generates and saves a bar chart showing the count of images for each class.

        Args:
            data (pd.DataFrame): DataFrame containing the image data with a 'name' column representing class names.
            save_path (str): Path where the generated graph will be saved.
            graph_title (str): Title of the graph.

        Returns:
            None: Saves the graph as an image file at the specified path.
        """
        data["name"].value_counts().plot(kind="bar")
        plt.title(graph_title)
        plt.xlabel("Mushroom Name")
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(save_paths)
        plt.close()

    @staticmethod
    def create_model_graph(fit_history: History) -> None:
        """
        Generates and saves graphs showing the training history of a model,
        including accuracy and loss for both training and validation data.

        Args:
            fit_history: The history object returned by the `model.fit` method,
                        containing training and validation metrics over epochs.

        Returns:
            None: Saves the generated graph as an image file.
        """
        plt.figure(1, figsize=(15, 8))

        plt.subplot(221)
        plt.plot(fit_history.history["accuracy"])
        plt.plot(fit_history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "valid"])

        plt.subplot(222)
        plt.plot(fit_history.history["loss"])
        plt.plot(fit_history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "valid"])

        plt.tight_layout()

        plt.savefig(Path.graphs_model_train_history)
        plt.close()
