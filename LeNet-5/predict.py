import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt


number_classes = 10

if __name__ == '__main__':
    print("CUDA Available: ", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=v2.Compose(
        [v2.Resize((32, 32)), v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
         v2.Normalize(mean=(0.1325,), std=(0.3105,))]), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    model = torch.load('models/mnist_latest.pkl', map_location=device)
    model.eval()
    for (i, (images, labels)) in enumerate(test_loader):
        images = images.to(device)
        if i > 3:
            break
        outputs = model.forward(images)
        value, predicted = torch.max(outputs.data, 1)
        plt.figure(i)
        plt.imshow(images.cpu()[0][0])
        plt.title("Prediction: " + str(int(predicted)))
    plt.show()

