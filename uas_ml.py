# -*- coding: utf-8 -*-
"""UAS ML

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qQFR5TPwOu8lLkB5Id-CZYPN59-yRuzI

#UAS Machine Learning
# **Kelompok 6**
1. Bayu Unggul Sejati
2. Hengky Triyo
3. Salwa Salsabila
4. Rafi Albar Kurniawan
"""


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import os
import flask

data_url = 'https://github.com/bay195/Waste-Classification/raw/main/Anorganik.zip'
urllib.request.urlretrieve(data_url, 'Anorganik.zip')
local_file = 'Anorganik.zip'
zip_ref = zipfile.ZipFile(local_file, 'r')
zip_ref.extractall('data/')
zip_ref.close()

BASE_DIR = 'data/Anorganik'
train_dir = os.path.join(BASE_DIR, 'Train')
validation_dir = os.path.join(BASE_DIR, 'Validation')

# Define image generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Use the correct target size for EfficientNet (e.g., 224x224 for EfficientNetB0)
target_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=32,
    class_mode='categorical',
    target_size=target_size
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    batch_size=32,
    class_mode='categorical',
    target_size=target_size
)

# Load the EfficientNetB0 model without the top classification layer
base_model = EfficientNetB1(include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Create a new model on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(6, activation='softmax')  # For multi-class classification with 6 categories
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
              metrics=['accuracy'])

# Define the callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='min'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,
    callbacks=callbacks
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

model.save("efficiennetB1.h5")

model = load_model("efficiennetB1.h5")

# Load and preprocess uploaded image
img_path = 'test5.jpg'  # Ganti dengan path gambar yang diunggah
img = image.load_img(img_path, target_size=(224, 224))  # Sesuaikan dengan ukuran input model Anda
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Use preprocess_input function for EfficientNet
img_array = preprocess_input(img_array)

# Predict class
prediction = model.predict(img_array)

# Display result
print(prediction)

# Memperoleh nama folder (kelas) dari generator pelatihan
class_names_train = train_generator.class_indices
print("Nama folder (kelas) dari generator pelatihan:", class_names_train)

# Memperoleh nama folder (kelas) dari generator validasi
class_names_validation = validation_generator.class_indices
print("Nama folder (kelas) dari generator validasi:", class_names_validation)

# Kamus yang memetakan indeks ke nama folder kategori
class_names = {'BotolKaca': 0, 'BotolPlastik': 1, 'GelasDisposable': 2, 'Kaleng': 3, 'Kardus': 4, 'WadahKaca': 5}

# Output probabilitas dari model
output_probabilities = prediction

# Konversi output probabilitas ke dalam array NumPy
output_probabilities_array = np.array(output_probabilities)

# Mendapatkan indeks dari kelas dengan probabilitas tertinggi
predicted_class_index = np.argmax(output_probabilities_array)

# Mendapatkan nama folder kategori berdasarkan indeks
predicted_class_name = [name for name, index in class_names.items() if index == predicted_class_index][0]

# Tampilkan nama folder kategori yang diprediksi
print("Folder kategori yang diprediksi:", predicted_class_name)

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training the model (fine-tuning)
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,  # Additional epochs for fine-tuning
    callbacks=callbacks
)

model.save("finedefficientnetB0.h5")

# Load and preprocess uploaded image
img_path = 'test1.jpg'  # Ganti dengan path gambar yang diunggah
img = image.load_img(img_path, target_size=(224, 224))  # Sesuaikan dengan ukuran input model Anda
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Use preprocess_input function for EfficientNet
img_array = preprocess_input(img_array)

# Predict class
prediction = model.predict(img_array)

# Display result
print(prediction)

# Kamus yang memetakan indeks ke nama folder kategori
class_names = {'BotolKaca': 0, 'BotolPlastik': 1, 'GelasDisposable': 2, 'Kaleng': 3, 'Kardus': 4, 'WadahKaca': 5}

# Output probabilitas dari model
output_probabilities = prediction

# Konversi output probabilitas ke dalam array NumPy
output_probabilities_array = np.array(output_probabilities)

# Mendapatkan indeks dari kelas dengan probabilitas tertinggi
predicted_class_index = np.argmax(output_probabilities_array)

# Mendapatkan nama folder kategori berdasarkan indeks
predicted_class_name = [name for name, index in class_names.items() if index == predicted_class_index][0]

# Tampilkan nama folder kategori yang diprediksi
print("Folder kategori yang diprediksi:", predicted_class_name)

# Load the best model
model.load_weights("best_model.h5")