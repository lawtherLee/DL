import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import pandas as pd
from sklearn.model_selection import train_test_split


def create_dataset():
    df = pd.read_csv("../data/手写数字识别.csv")
    x = df.iloc[:, 1:].values / 255
    y = df.iloc[:, 0].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    x_train = torch.tensor(x_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
    x_test = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.5)

        self.linear1 = nn.Linear(20 * 4 * 4, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, _x):
        x = torch.relu(self.conv1(_x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # print(_x.shape)
        # print(x.shape)
        # print(x.size())
        x = x.reshape(x.size(0), -1)
        # print(x.size())
        x = self.dropout(x)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


def train(train_loader, model, criterion, optimizer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(test_loader, model, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


if __name__ == "__main__":
    epochs = 10
    train_loader, test_loader = create_dataset()

    # for images, labels in train_loader:
    #     print(images.shape)
    #     print(labels.shape)
    #     break
    model = DigitCNN()
    # summary(model, (1, 28, 28), batch_size=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        test_loss, test_acc = evaluate(test_loader, model, criterion)
        print(
            f"epoch: {epoch + 1}, "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "../model/mnist_cnn.pth")
            print(f"保存最佳模型，best_acc: {best_acc:.4f}")
    print(f"训练结束，最佳准确率：{best_acc:.4f}")
