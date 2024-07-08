import torch.nn as nn
import torch.nn.functional as Functions

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.input_layer = 
        self.hidden_layer = 
        self.output_layer = 
    
    def forward(self, input_data):
        input_data = input_data.view(-1, 28*28)  # Flatten the input
        output =   # Input layer + ReLU
        output =   # Hidden layer + ReLU
        output =   # Output layer (logits)
        return output

# Instantiate and print the model
model = SimpleNN()
print(model)