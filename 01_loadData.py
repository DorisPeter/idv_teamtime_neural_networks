import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformationen definieren
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# MNIST-Daten laden
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader erstellen
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Daten erfolgreich geladen.")

# Anzahl der Datensätze im Trainingsdatensatz ermitteln
num_train_samples = len(train_dataset)
print(f"Anzahl der Datensätze im Trainingsdatensatz: {num_train_samples}")
# Anzahl der Datensätze im Trainingsdatensatz ermitteln
num_test_samples = len(test_dataset)
print(f"Anzahl der Datensätze im Testdatensatz: {num_test_samples}")