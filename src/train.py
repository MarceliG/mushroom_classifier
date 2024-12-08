import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Train:

    @staticmethod
    def prepare(data_path: str):
        training_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_generator = training_datagen.flow_from_directory(
            data_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        valid_generator = training_datagen.flow_from_directory(
            data_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )

        return (
            train_generator,
            valid_generator
        )
    
    @staticmethod
    def build_model():
        return models.Sequential([
            layers.Input(shape=(150, 150, 3)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            # Flatten the result to feed into a DNN
            layers.Flatten(),
            # layers.Dropout(0.5),
            # 512 neuron hidden layer
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(9, activation='softmax')
        ])

    @staticmethod
    def execute(data_path: str, save_path: str) -> None:
        (train_dataset, validate_dataset) = Train.prepare(data_path)

        model = Train.build_model()

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            save_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

        model.fit(
            train_dataset,
            epochs=10,
            validation_data=validate_dataset,
            callbacks=[early_stopping, checkpoint],
            verbose=1,
            # use_multiprocessing=True,
            # workers=4,
            # max_queue_size=20
        )

        model.save(save_path, save_format='tf')

