import torch
from torch import nn
from torch import optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from MuVAN.dataset import load_mhealth
from MuVAN.model import MuVANminus


(x_train, y_train), (x_test, y_test) = load_mhealth(256, 0.9)

y_train -= 1
y_test -= 1

train_ds = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
train_loader = DataLoader(train_ds, batch_size=32)


test_ds = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).long())
test_loader = DataLoader(test_ds, batch_size=32)

model = MuVANminus(64, 1856)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)


def train(load=True):
    if load:
        if load:
            model.load_state_dict(torch.load("minus.pth"))

    model.train()
    for epoch in range(2):
        for s, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x)

            loss = loss_func(output, y[:, 0])

            loss.backward()
            optimizer.step()
            if s % 100 == 0:
                print("-"*50)

        print(epoch)


def eval(load=True):
    if load:
        model.load_state_dict(torch.load("minus.pth"))

    running_loss = torch.tensor(0.0).float()
    running_corrects = torch.tensor(0).long()
    len_of_dataset = len(test_loader.dataset)

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)

            _, predictions = torch.max(output, 1)
            loss = loss_func(output, y[:, 0])

            running_loss += loss.item() * x.shape[0]
            running_corrects += torch.sum(predictions == y[:, 0].data)

    test_loss = running_loss / len_of_dataset
    test_acc = running_corrects.float() / len_of_dataset
    print("Loss: {:.3f} - Accuracy: {:.3%}".format(test_loss, test_acc))


def main():
    train()
    eval()
    torch.save(model.state_dict(), "minus.pth")


if __name__ == '__main__':
    main()
