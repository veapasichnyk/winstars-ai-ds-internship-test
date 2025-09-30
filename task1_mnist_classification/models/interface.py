from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """Abstract interface for MNIST classifiers."""

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict labels for given input data."""
        pass