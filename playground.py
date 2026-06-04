import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, output_dim)

    def forward(self, x):
        return torch.softmax(self.linear2(torch.relu(self.linear1(x))), dim=1)


def train(data):
    model = Model(20, 5)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    dataloader = DataLoader(data, batch_size=8, shuffle=True)
    epochs = 20

    for epoch in range(epochs):
        total_loss = 0
        total_num = 0
        model.train()
        for x, y in dataloader:
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += len(y)
            total_loss += loss.item() * len(y)
        print(f"epoch: {epoch + 1:4} loss: {total_loss / total_num:.2f}")
    torch.save(model.state_dict(), "./model/train_data.pth")


def create_dataset():
    data = pd.read_csv("./data/手机价格预测.csv")
    x = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(data.iloc[:, -1].values, dtype=torch.long)
    y = F.one_hot(y, num_classes=5).to(torch.float32)
    return TensorDataset(x, y)


if __name__ == "__main__":
    train_data = create_dataset()
    train(train_data)
