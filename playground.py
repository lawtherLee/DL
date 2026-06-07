from sklearn.externals.array_api_compat import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from torchsummary import summary


def create_dataset():
    train_dataset = CIFAR10(
        root="./data",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_dataset = CIFAR10(
        root="./data",
        train=False,
        transform=ToTensor(),
        download=True,
    )
    return train_dataset, test_dataset


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3,
            6,
            3,
        )
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(
            6,
            16,
            3,
        )
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        _x = self.pool2(torch.relu(self.conv2(self.pool1(torch.relu(self.conv1(x))))))
        # print(x.shape)
        # print(_x.shape)
        # print(_x.reshape(_x.size(0), -1).shape)
        return self.output(
            torch.relu(
                self.linear2(torch.relu(self.linear1(_x.reshape(_x.size(0), -1))))
            )
        )


def train(train_dataset):
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    model = ImageModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0, 0, 0, time.time()
        for x, y in dataloader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(torch.argmax(y_pred, dim=-1))
            # print(y)
            # print(torch.argmax(y_pred, dim=-1) == y)
            # print((torch.argmax(y_pred, dim=-1) == y).sum())
            total_loss += loss.item() * len(y)
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_samples += len(y)

        print(
            f"epoch{epoch + 1}, loss: {total_loss / total_samples:.4f}, acc: {total_correct / total_samples:.2f}, time: {time.time() - start}s"
        )
    torch.save(model.state_dict(), "./model/image_model.pth")


def evaluate(test_dataset):
    pass


if __name__ == "__main__":
    batch_size = 8
    train_dataset, test_dataset = create_dataset()
    # print(f"训练集: {train_dataset.data.shape}")
    # model = ImageModel()
    # summary(model, (3, 32, 32), batch_size)

    train(train_dataset)

    evaluate(test_dataset)
