import torch.nn as nn
import torch.nn.functional as Functions

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.input_layer = nn.Linear(28*28, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)
    
    def forward(self, input_data):
        input_data = input_data.view(-1, 28*28)  # Flatten the input
        output = Functions.relu(self.input_layer(input_data))  # Input layer + ReLU
        output = Functions.relu(self.hidden_layer(output))  # Hidden layer + ReLU
        output = self.output_layer(output)  # Output layer (logits)
        return output

# Instantiate and print the model
model = SimpleNN()
print(model)