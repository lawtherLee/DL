import torch
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

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.linear1 = nn.Linear(128 * 8 * 8, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        self.output = nn.Linear(2048, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.reshape(x.size(0), -1)

        x = torch.relu(self.linear1(x))
        x = self.dropout(x)

        x = torch.relu(self.linear2(x))
        x = self.dropout(x)

        return self.output(x)


def train(train_dataset):
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    model = ImageModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    epochs = 50
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0, 0, 0, time.time()
        model.train()

        for x, y in dataloader:
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
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    model = ImageModel()
    model.load_state_dict(torch.load("./model/image_model.pth"))
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x)
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_samples += len(y)

    print(f"acc: {total_correct / total_samples:.2f}")


if __name__ == "__main__":
    batch_size = 128
    train_dataset, test_dataset = create_dataset()
    # print(f"训练集: {train_dataset.data.shape}")
    # model = ImageModel()
    # summary(model, (3, 32, 32), batch_size)

    train(train_dataset)

    evaluate(test_dataset)
