import torchvision
import torchvision.transforms as transforms
import torch

def load_mnist():
    """
    Load the MNIST dataset with torchvision.
    
    Returns:
        trainset (Dataset): Training part of MNIST
        testset (Dataset): Test part of MNIST
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return trainset, testset


def prepare_for_rf(trainset, testset):
    """
    Prepare data for sklearn models (Random Forest).
    Flatten images into 2D arrays.

    Args:
        trainset (Dataset): MNIST training dataset
        testset (Dataset): MNIST test dataset
    
    Returns:
        X_train, y_train, X_test, y_test (numpy arrays)
    """
    X_train = trainset.data.view(-1, 28*28).numpy()
    y_train = trainset.targets.numpy()
    X_test = testset.data.view(-1, 28*28).numpy()
    y_test = testset.targets.numpy()
    return X_train, y_train, X_test, y_test


def prepare_for_torch(trainset, testset):
    """
    Prepare data as Tensors for PyTorch models (NN, CNN).
    Normalize values to [0, 1].

    Args:
        trainset (Dataset): MNIST training dataset
        testset (Dataset): MNIST test dataset
    
    Returns:
        X_train, y_train, X_test, y_test (torch tensors)
    """
    X_train = trainset.data.unsqueeze(1).float() / 255.0
    y_train = trainset.targets
    X_test = testset.data.unsqueeze(1).float() / 255.0
    y_test = testset.targets
    return X_train, y_train, X_test, y_test


def get_dataloaders(X_train, y_train, batch_size=64):
    """
    Create DataLoader from tensors.

    Args:
        X_train (Tensor): Training features
        y_train (Tensor): Training labels
        batch_size (int): Batch size
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader