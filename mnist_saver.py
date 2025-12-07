import numpy as np
from tensorflow.keras.datasets import mnist

# Загружаем данные
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Сохраняем в бинарном формате
with open('data/train-images.bin', 'wb') as f:
    f.write(train_images.astype('uint8').tobytes())
with open('data/train-labels.bin', 'wb') as f:
    f.write(train_labels.astype('uint8').tobytes())
with open('data/test-images.bin', 'wb') as f:
    f.write(test_images.astype('uint8').tobytes())
with open('data/test-labels.bin', 'wb') as f:
    f.write(test_labels.astype('uint8').tobytes())

print(f'Сохранено {len(train_images)} обучающих и {len(test_images)} тестовых изображений')