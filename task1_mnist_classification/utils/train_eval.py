import torch
from sklearn.metrics import accuracy_score

def train_torch_model(model, dataloader, optimizer, criterion, device, epochs=5):
    """
    Train a PyTorch model for the specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer instance
        criterion (torch.nn.Module): Loss function
        device (str): 'cpu' or 'cuda'
        epochs (int): Number of epochs
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")


def evaluate_torch_model(model, X_test, y_test, device):
    """
    Evaluate a PyTorch model on test data using accuracy metric.

    Args:
        model (torch.nn.Module): Trained model
        X_test (Tensor): Test features
        y_test (Tensor): Test labels
        device (str): 'cpu' or 'cuda'
    
    Returns:
        acc (float): Accuracy
        preds (Tensor): Predicted labels
    """
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        preds = outputs.argmax(dim=1).cpu()
    acc = accuracy_score(y_test.cpu(), preds)
    return acc, preds