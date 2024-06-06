import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuração do logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega o modelo treinado
model = load_model('trained_model.h5')
logging.info("Modelo carregado.")

# Carrega o encoder de rótulos
class_indices = np.load('label_encoder.npy', allow_pickle=True).item()
classes = {v: k for k, v in class_indices.items()}
logging.info("Encoder de rótulos carregado.")

# Caminho para os dados de teste
test_path = './dataset/data/testing_data'
batch_size = 16
image_size = (64, 64)

# Data Generator para teste
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Avalia o modelo
logging.info("Avaliando o modelo.")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")
logging.info(f"Acurácia do teste: {test_accuracy}")

# Faz predições
logging.info("Iniciando predições.")
preds = model.predict(test_generator)
predicted_labels = np.argmax(preds, axis=1)
logging.info("Predições concluídas.")

# Seleciona uma imagem aleatória para avaliação
# Índice aleatório do lote
batch_index = random.randint(0, len(test_generator) - 1)
# Obtém o lote de imagens e rótulos
image_batch, label_batch = test_generator[batch_index]
# Índice aleatório da imagem no lote
image_index = random.randint(0, len(image_batch) - 1)

random_image = image_batch[image_index]
random_label = label_batch[image_index]
predicted_label = predicted_labels[batch_index * batch_size + image_index]

# Mostra a imagem aleatória com seu rótulo previsto
plt.imshow(random_image)
plt.title(f"Verdadeiro: {classes[int(random_label)]}, Previsto: {
          classes[predicted_label]}")
plt.show()
logging.info(f"Imagem aleatória rotulada como: Verdadeiro: {
             classes[int(random_label)]}, Previsto: {classes[predicted_label]}")
