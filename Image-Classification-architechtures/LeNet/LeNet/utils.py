import torch
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

def download_data():
    # define transforms
    transforms = transforms.Compose([transforms.Resize(32,32),
                 transforms.ToTensor()])
    train_data = MNIST(root='/', train=True, transform=transforms,download=True)
    test_data = MNIST(root='/', train=False, transform=transforms, download=True)

    