from .rf_classifier import RandomForestClassifierMNIST
from .nn_classifier import FeedForwardNN
from .cnn_classifier import CNNClassifier

class MnistClassifier:
    """Wrapper to select and run different MNIST classifiers."""
    def __init__(self, algorithm="rf", **kwargs):
        if algorithm == "rf":
            self.model = RandomForestClassifierMNIST(**kwargs)
        elif algorithm == "nn":
            self.model = FeedForwardNN(**kwargs)
        elif algorithm == "cnn":
            self.model = CNNClassifier(**kwargs)
        else:
            raise ValueError("Unknown algorithm. Choose from: 'rf', 'nn', 'cnn'.")

    def train(self, X_train, y_train):
        return self.model.train(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)