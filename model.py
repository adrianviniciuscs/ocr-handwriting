import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Configuração do logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Caminho para os dados de treinamento
train_path = "./dataset/data/training_data"
batch_size = 16
image_size = (64, 64)

# Data Generator para treinamento e validação
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# Constrói o modelo
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])
logging.info("Modelo construído.")

# Compila o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
logging.info("Modelo compilado.")

# Treina o modelo
logging.info("Iniciando treinamento do modelo.")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)
logging.info("Treinamento concluído.")

# Salva o modelo e o encoder
model.save('trained_model.h5')
np.save('label_encoder.npy', train_generator.class_indices)
logging.info("Modelo e encoder salvos.")

