# Task 1 — MNIST Classification (Random Forest, NN, CNN)

## **Project Description**

This task demonstrates image classification on the MNIST dataset using three different approaches:

Random Forest (sklearn)

Feed-Forward Neural Network (PyTorch)

Convolutional Neural Network (CNN) (PyTorch)

All models are implemented as classes that follow the same OOP interface (MnistClassifierInterface), and are wrapped by a manager class (MnistClassifier).
This ensures that training and inference APIs remain consistent regardless of the chosen algorithm.

## **Project Structure**
```text
│
├── task1_mnist_classification/
│ │── README.md # Documentation for Task 1
│ │── requirements.txt # Dependencies for Task 1 only
│ │── demo.ipynb # Jupyter Notebook with model training & demo
│ │
│ ├── models/
│ │ │── interface.py # Abstract base class (MnistClassifierInterface)
│ │ │── rf_classifier.py # Random Forest implementation
│ │ │── nn_classifier.py # Feed-Forward NN implementation
│ │ │── cnn_classifier.py # CNN implementation
│ │ │── classifier.py # Wrapper class (MnistClassifier)
│ │
│ └── utils/
│ │── data_loader.py # Data loading and preprocessing helpers
│ │── train_eval.py # Training and evaluation functions
│
```

## **Setup**

Clone the repository:

git clone https://github.com/your-username/winstars-ai-ds-internship-test.git
cd winstars-ai-ds-internship-test/task1_mnist_classification


Install dependencies:

pip install -r requirements.txt


Run the demo:

Open demo.ipynb in Google Colab or locally with Jupyter Notebook.

The notebook will train and evaluate Random Forest, Feed-Forward NN, CNN, and show sample predictions.

**Example Results**
```
Accuracy:
Random Forest   : 0.9705
Feed-Forward NN : 0.9618
CNN             : 0.9879
```

**Sample Predictions:**
```
True	RF	NN	 CNN
0	    0  	 0	   0
7	    7	 7	   7
6	    1	 1	   6 
```
Observation: CNN is the most robust model, correctly classifying ambiguous digits (e.g., distinguishing 6 vs 1).

## **Conclusion**

Random Forest provided a strong baseline with ~97% accuracy.

Feed-Forward NN achieved ~96%, slightly lower due to its limited ability to extract spatial features.

CNN achieved the best result (~99%), confirming that convolutional architectures are the most effective for image classification tasks.
