import torch
import torch.nn as nn
from torchsummary import summary


class ModelDemo(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)

        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, _x):
        return torch.softmax(
            self.output(torch.relu(self.linear2(torch.sigmoid(self.linear1(_x))))),
            dim=1,
        )


def train():
    my_model = ModelDemo()
    # print(my_model)

    data = torch.randn(5, 3)
    print(f"data: {data}")
    print(f"data.shape: {data.shape}")
    print(f"data.requires_grad: {data.requires_grad}")

    output = my_model(data)
    print(f"output: {output}")
    print(f"output.shape: {output.shape}")
    print(f"output.requires_grad: {output.requires_grad}")
    # print("*" * 50)

    summary(my_model, (5, 3))

    for name, param in my_model.named_parameters():
        print(f"name: {name}")
        print(f"param: {param}\n")


if __name__ == "__main__":
    train()
