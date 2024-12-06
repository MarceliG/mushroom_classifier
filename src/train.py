from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input

from src.resources import Preprocess

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Train:
    img_size = 150 # docelowo na 300

    @staticmethod
    def prepare(data_path: str):
        (train_data, val_data, test_data) = Preprocess(data_path).execute()

        train_gen = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            shear_range=10,
            zoom_range=0.1,
            vertical_flip=True
        )

        train_ds = train_gen.flow_from_dataframe(
            train_data,
            directory=None,
            x_col='filepath',
            y_col='name',
            target_size=(Train.img_size, Train.img_size),
            batch_size=32
        )

        val_gen = ImageDataGenerator(preprocessing_function = preprocess_input)

        val_ds = val_gen.flow_from_dataframe(
            val_data,
            directory=None,
            x_col='filepath',
            y_col='name',
            target_size=(Train.img_size, Train.img_size),
            batch_size=32,
            shuffle=False
        )

        return (
            train_ds,
            val_ds
        )
    
    @staticmethod
    def build_model(
        learning_rate = 0.01, 
        size_inner=1000, 
        size_inner_one=100, 
        size_inner_two=100
    ):

        model = EfficientNetV2B0(
            weights='imagenet',
            include_top = False,
            input_shape=(Train.img_size, Train.img_size, 3)
        )

        model.trainable = False

        inputs = keras.Input(shape=(Train.img_size, Train.img_size, 3))
        base = model(inputs, training=False)
        vectors = keras.layers.GlobalAveragePooling2D()(base)

        inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
        inner_one = keras.layers.Dense(size_inner_one, activation='relu')(inner)
        inner_two = keras.layers.Dense(size_inner_two, activation='relu')(inner_one)

        outputs = keras.layers.Dense(9)(inner_two)
        model = keras.Model(inputs, outputs)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = keras.losses.CategoricalCrossentropy(from_logits=True) # because multiclass classification
        
        model.compile(
            optimizer=optimizer, 
            loss=loss, 
            metrics=['accuracy']
        )
        
        return model

    @staticmethod
    def execute(mushrooms_path: str, save_path: str) -> None:
        (train_ds, val_ds) = Train.prepare(mushrooms_path)

        model = Train.build_model()

        model.fit(train_ds, epochs=10, validation_data=val_ds)

        model.save(save_path, save_format='tf')

