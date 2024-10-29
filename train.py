import os
import zipfile
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Paths to your zip files
zip_path_1 = 'HAM10000_images_part_1.zip'
zip_path_2 = 'HAM10000_images_part_2.zip'

# Load the metadata
df = pd.read_csv('HAM10000_metadata.csv')

# Check the number of unique classes
num_classes = df['dx'].nunique()
print(f"Number of unique classes: {num_classes}")

# Function to get image from ZIP
def get_image_from_zip(image_id):
    image_file = f'{image_id}.jpg'

    with zipfile.ZipFile(zip_path_1, 'r') as z1:
        if image_file in z1.namelist():
            with z1.open(image_file) as image_data:
                return Image.open(BytesIO(image_data.read()))

    with zipfile.ZipFile(zip_path_2, 'r') as z2:
        if image_file in z2.namelist():
            with z2.open(image_file) as image_data:
                return Image.open(BytesIO(image_data.read()))

    return None

# Preprocess and build the model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # num of unique classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model():
    model = build_model()

    # Split data into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['dx'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['dx'], random_state=42)

    # Function to generate images and labels from the dataframe
    def dataframe_to_image_label_generator(df, batch_size=32, target_size=(128, 128)):
        class_map = {label: index for index, label in enumerate(df['dx'].unique())}
        while True:
            for start in range(0, len(df), batch_size):
                end = min(start + batch_size, len(df))
                batch_df = df.iloc[start:end]
                images = []
                labels = []
                for _, row in batch_df.iterrows():
                    img = get_image_from_zip(row['image_id']).resize(target_size)
                    img = np.array(img) / 255.0
                    images.append(img)
                    labels.append(class_map[row['dx']])
                images = np.array(images)
                labels = to_categorical(np.array(labels), num_classes=num_classes)
                yield images, labels

    # Create generators
    train_gen = dataframe_to_image_label_generator(train_df)
    val_gen = dataframe_to_image_label_generator(val_df)

    # Train the model
    model.fit(train_gen, steps_per_epoch=len(train_df) // 32, validation_data=val_gen,
              validation_steps=len(val_df) // 32, epochs=15)

    # Save the trained model
    model.save('skin_cancer_detector.h5')

    print("Model saved as 'skin_cancer_detector.h5'")

if __name__ == "__main__":
    train_model()
