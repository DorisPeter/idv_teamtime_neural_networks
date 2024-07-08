import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTModel:
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 5

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.train_loader, self.test_loader = self.load_data()
        self.model = self.build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_data(self):
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

    def build_model(self):
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.input_layer = nn.Linear(28*28, 128)
                self.hidden_layer = nn.Linear(128, 64)
                self.output_layer = nn.Linear(64, 10)
            
            def forward(self, input_data):
                input_data = input_data.view(-1, 28*28)  # Flatten the input
                output = torch.relu(self.input_layer(input_data))  # Input layer + ReLU
                output = torch.relu(self.hidden_layer(output))  # Hidden layer + ReLU
                output = self.output_layer(output)  # Output layer (logits)
                return output

        return SimpleNN()

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}")
        print("Training finished.")
    
    def evaluate(self):
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

    def save_model(self, path='simple_nn.pth'):
        torch.save(self.model.state_dict(), path)
        print("Saved model.")

    def load_model(self, path='simple_nn.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print("loaded model.")
    

# Beispielhafte Nutzung der Klasse
if __name__ == '__main__':
    mnist_model = MNISTModel()
    mnist_model.train()
    mnist_model.evaluate()
    mnist_model.save_model()
    mnist_model.load_model()