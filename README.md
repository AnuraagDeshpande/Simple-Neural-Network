# Simple Neural Network for Iris Flower Classification

This project implements a simple feedforward neural network using PyTorch to classify iris flower species based on their features. The dataset used is the popular Iris dataset, which contains measurements for three classes of iris flowers: Setosa, Versicolor, and Virginica.

---

## Project Overview
The implemented neural network architecture is as follows:
1. Input layer: Accepts 4 features (sepal length, sepal width, petal length, petal width).
2. Hidden layers:
   - Hidden Layer 1 (h1): 9 neurons.
   - Hidden Layer 2 (h2): 9 neurons.
3. Output layer: 3 neurons corresponding to the 3 classes of iris flowers.

The network uses ReLU as the activation function for the hidden layers and raw logits in the output layer. The loss is computed using CrossEntropyLoss, which internally applies softmax to the logits.

---

## Features and Functionality
- **Dataset Handling**:
  - The Iris dataset is loaded directly from an online source using pandas.
  - The target labels (flower classes) are converted from string values (Setosa, Versicolor, Virginica) to numeric values (0, 1, 2).
  - Features and labels are split into training and testing sets using `train_test_split` from `sklearn`.

- **Neural Network Architecture**:
  - Defined using PyTorch's `nn.Module`.
  - Includes forward propagation with two hidden layers and an output layer.

- **Training**:
  - Implements backpropagation using the Adam optimizer.
  - Tracks loss over multiple epochs.
  - Prints the loss every 10 epochs for monitoring.

---

## Code Structure
### Key Components:
1. **Dataset Preparation**:
   - Load data from the CSV file.
   - Preprocess the data (convert target labels, split into training and testing sets).
   - Convert the data into PyTorch tensors.

2. **Neural Network Definition**:
   - `Model` class inheriting from `nn.Module`.
   - Defines layers (`fc1`, `fc2`, `out`) and the `forward` method.

3. **Training Loop**:
   - Perform forward propagation.
   - Compute loss using `CrossEntropyLoss`.
   - Update weights using backpropagation.

4. **Visualization**:
   - Loss values can be visualized to monitor training progress.

---

## Requirements
To run this project, you will need the following:
- Python 3.7+
- Libraries:
  - `torch`
  - `torch.nn`
  - `pandas`
  - `numpy`
  - `sklearn`
  - `matplotlib`

Install the required libraries using pip:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

---

## How to Run
1. Clone this repository or copy the project files.
2. Run the Python script to train the model:
   ```bash
   python iris_nn.py
   ```
3. The training progress, including loss values, will be displayed in the terminal.

---

## Results
- The model achieves reasonable accuracy on the test data for a simple feedforward network.
- Loss decreases consistently during training, as displayed by periodic logs.

---

## Next Steps
To extend this project, consider:
- Implementing a more complex architecture (e.g., additional layers, different activation functions).
- Hyperparameter tuning (learning rate, number of neurons, epochs, etc.).
- Visualizing the decision boundaries using matplotlib.
- Saving and loading the trained model using `torch.save` and `torch.load`.

---

## Acknowledgments
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Iris Dataset](https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv)

