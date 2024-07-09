import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# MNIST-Datensatz laden
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Ein einzelnes Bild auswählen (zum Beispiel das erste Bild im Trainingsdatensatz)
image, label = mnist_train[1000]

# Das Bild anzeigen
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()

# Das Bild speichern
plt.imsave('mnist_image_1000.png', image.squeeze(), cmap='gray')