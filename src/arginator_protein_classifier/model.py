import torch
from torch import nn


class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""

    def __init__(self, input_dim, output_dim, dropout_rate) -> None:
        super().__init__()
        self.input_dim = (input_dim,)
        self.output_dim = (output_dim,)
        self.fc1 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)


if __name__ == "__main__":
    x = torch.rand(1, 1024)
    mock = Model(x.shape[0], 2, 0.3)
    output = mock.forward(x)
    print(output.shape)

# # Imagine this is your dataset (batch_size, feature_dim)
# # Case A: 1024-dimensional embeddings
# data_a = torch.randn(32, 1024)

# # Case B: 512-dimensional embeddings
# data_b = torch.randn(32, 512)

# # --- Dynamic Instantiation ---

# # 1. Get the feature dimension from the data
# # data.shape[1] gives the feature size (1024 or 512)
# input_dim_a = data_a.shape[1]

# # 2. Pass it to the model
# model = DynamicNet(input_size=input_dim_a, hidden_size=256, output_size=10)

# print(f"Model created with input size: {model.layer1.in_features}")
# # Output: Model created with input size: 1024
