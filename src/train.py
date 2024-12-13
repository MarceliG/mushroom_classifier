from PIL import ImageFile
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import logger
from src.graphs import Graphs
from src.resources import Preprocess

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Train:
    img_size = 300

    @staticmethod
    def prepare(data_path: str) -> tuple:
        """
        Prepares training, validation, and test datasets from the provided data path.

        Args:
            data_path (str): The path to the data directory or file containing training,
                            validation, and test data information.

        Returns:
            tuple: A tuple containing:
                - train_ds: The training dataset generator.
                - val_ds: The validation dataset generator.
                - test_ds: The test dataset generator.
        """
        (train_data, val_data, test_data) = Preprocess(data_path).execute()

        # Create train dataset
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input, shear_range=10, zoom_range=0.1, vertical_flip=True
        )

        train_ds = train_gen.flow_from_dataframe(
            train_data,
            directory=None,
            x_col="filepath",
            y_col="name",
            target_size=(Train.img_size, Train.img_size),
            batch_size=32,
        )

        # Create validate dataset
        val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        val_ds = val_gen.flow_from_dataframe(
            val_data,
            directory=None,
            x_col="filepath",
            y_col="name",
            target_size=(Train.img_size, Train.img_size),
            batch_size=32,
            shuffle=False,
        )

        # Create test dataset
        test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        test_ds = test_gen.flow_from_dataframe(
            test_data,
            directory=None,
            x_col="filepath",
            y_col="name",
            target_size=(Train.img_size, Train.img_size),
            batch_size=32,
            shuffle=False,
        )

        return (train_ds, val_ds, test_ds)

    @staticmethod
    def build_model(
        learning_rate: float = 0.001,
        size_inner: int = 1000,
        size_inner_one: int = 1000,
        size_inner_two: int = 100,
        droprate: float = 0.2,
    ) -> keras.Model:
        """
        Builds and compiles a neural network model using EfficientNetV2B0 as the base.

        Args:
            learning_rate (float): Learning rate for the Adam optimizer.
            size_inner (int): Number of neurons in the first dense layer.
            size_inner_one (int): Number of neurons in the second dense layer.
            size_inner_two (int): Number of neurons in the third dense layer.
            droprate (float): Dropout rate for regularization in the dropout layer.

        Returns:
            keras.Model: A compiled Keras model.
        """
        model = EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(Train.img_size, Train.img_size, 3))

        model.trainable = False

        inputs = keras.Input(shape=(Train.img_size, Train.img_size, 3))
        base = model(inputs, training=False)
        vectors = keras.layers.GlobalAveragePooling2D()(base)

        # Add custom fully connected layers.
        inner = keras.layers.Dense(size_inner, activation="relu")(vectors)
        inner_one = keras.layers.Dense(size_inner_one, activation="relu")(inner)
        inner_two = keras.layers.Dense(size_inner_two, activation="relu")(inner_one)

        drop = keras.layers.Dropout(droprate)(inner_two)

        outputs = keras.layers.Dense(9)(drop)
        model = keras.Model(inputs, outputs)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)  # because multiclass classification

        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        return model

    @staticmethod
    def execute(mushrooms_path: str, save_path: str) -> None:
        """
        Executes the training pipeline for a model with the provided dataset and saves the model.

        Args:
            mushrooms_path (str): Path to the dataset containing training, validation, and test data.
            save_path (str): Path to save the trained model.

        Returns:
            None
        """
        (train_ds, val_ds, test_ds) = Train.prepare(mushrooms_path)

        model = Train.build_model()

        early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

        checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor="val_accuracy", mode="max")

        fit_history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[early_stopping, checkpoint])

        model.save(save_path)
        Graphs.create_model_graph(fit_history)

        # Testing the model on test data it hasn't seen yet
        logger.info("Test model on data it doesnt see")
        model.evaluate(test_ds)
