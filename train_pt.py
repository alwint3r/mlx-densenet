import torch
import torch.nn as nn
from torchvision import datasets
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from densenet.pytorch import DenseNet, BasicBlock


def train(epoch, net, device, trainloader, optimizer, criterion):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 200 == 199:  # Print every 200 mini-batches
            print(
                "[%d, %5d] loss: %.3f, accuracy: %.3f"
                % (epoch + 1, i + 1, running_loss / 200, 100 * correct / total)
            )
            running_loss = 0.0
            correct = 0
            total = 0


def evaluate(net, testloader, device):
    net.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    accuracy = (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    return precision, recall, f1, accuracy


def main():
    # Define the transformations for the CIFAR-100 dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ],
    )

    # Download the CIFAR-100 training and test datasets
    trainset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
    )

    testset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=2,
    )

    # Initialize the DenseNet model
    net = DenseNet(
        BasicBlock, [6, 12, 24, 16], num_classes=100
    )  # CIFAR-100 has 100 classes
    device = torch.device("mps" if torch.mps else "cpu")
    net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1):
        train(epoch, net, device, trainloader, optimizer, criterion)

    # Call the evaluation function after training
    precision, recall, f1, accuracy = evaluate(net, testloader, device)

    # Print the results
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))
    print("Accuracy: {:.4f}".format(accuracy))


if __name__ == "__main__":
    main()
