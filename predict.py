import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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

# Configura o parser de argumentos
parser = argparse.ArgumentParser(description='Run the OCR model on an image.')
parser.add_argument('image', type=str, nargs=1,
                    help='path of the image to run the prediction')

args = parser.parse_args()
image_path = args.image[0]
logging.info(f"Caminho da imagem: {image_path}")

# Carrega e pré-processa a imagem
img = image.load_img(image_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Faz predições
logging.info("Iniciando predições.")
pred = model.predict(img_array)
predicted_label = np.argmax(pred, axis=1)[0]
logging.info("Predições concluídas.")

# Mostra a imagem com seu rótulo previsto
plt.imshow(image.load_img(image_path, target_size=(64, 64)))
plt.title(f"Previsto: {classes[predicted_label]}")
plt.show()
logging.info(f"Resultado da predição: {classes[predicted_label]}")

