from model import Net
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

batch_size = 64
number_classes = 10
learning_rate = 0.001
num_epochs = 10

if __name__ == '__main__':
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    writer = SummaryWriter()

    transform = v2.Compose([v2.Resize((32, 32)), v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                            v2.Normalize(mean=(0.1325,), std=(0.3105,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = Net(number_classes).to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    last_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)
            loss = cost(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Epoch {}: loss/train'.format(epoch), loss.item(), i)
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total

        writer.add_scalar('Accuracy/train'.format(epoch), accuracy, epoch)
        print('Accuracy: {:.3f} %'.format(accuracy * 100))
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(accuracy * 100))
        if abs(accuracy - last_accuracy) < 1e-4:
            last_accuracy = accuracy
            break
        last_accuracy = accuracy

    if os.path.exists('models/mnist_latest.pkl'):
        os.remove('models/mnist_latest.pkl')
    os.rename('models/mnist_{:.3f}.pkl'.format(last_accuracy * 100), 'models/mnist_latest.pkl')
    print("Model finished training")
    writer.close();
