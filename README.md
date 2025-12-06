MNIST Neural Network from Scratch (NumPy Only)

📌 Overview

This project is a complete implementation of a Deep Neural Network built entirely from scratch using Python and NumPy.

Unlike standard projects that rely on high-level frameworks like TensorFlow or PyTorch, this implementation manually handles:

Matrix Calculus & Linear Algebra

Forward & Backward Propagation

Gradient Descent Optimization

Final Accuracy: ~85-90% on MNIST Test Set.

🧠 The Architecture

The network consists of an input layer, two hidden layers (using a mix of ReLU and Sigmoid activations), and an output layer using Softmax.

Layer

Nodes

Activation

Initialization

Input

784

N/A

N/A

Hidden 1

128

ReLU

He Initialization

Hidden 2

32

Sigmoid

He Initialization

Output

10

Softmax

He Initialization

🔧 Key Implementation Details

1. Vectorized Operations

To ensure efficient training on the CPU, all operations are vectorized using NumPy broadcasting. This replaces slow Python loops with highly optimized C-level matrix operations.

2. Hybrid Activation & Backpropagation

I implemented a mixed architecture to handle non-linearity:

ReLU (Layer 1): Used to prevent the vanishing gradient problem in the earlier layers.

Sigmoid (Layer 2): Used to squash features into a bounded range before the final classification.

Derivatives: Manually derived the gradients for both functions ($g'(z) = 1$ for $z>0$ and $g'(z) = a(1-a)$) to enable correct backpropagation.

3. Stability Mechanics

Softmax: Implemented with numerical stability shifts (subtracting max) to prevent exponential overflow.

He Initialization: Applied scaled random initialization ($\sqrt{\frac{2}{n_{in}}}$) to stabilize variance, particularly for the ReLU layer.

📉 Results

Training over X epochs with a learning rate of 0.1:

(Insert your accuracy graph screenshot here)

🚀 How to Run

Clone the repo:

git clone [https://github.com/yourusername/mnist-from-scratch.git](https://github.com/yourusername/mnist-from-scratch.git)


Install dependencies:

pip install numpy pandas matplotlib scikit-learn


Run the notebook:

jupyter notebook MNIST_NN_Scratch.ipynb


📚 References

Input Data: MNIST Dataset

Mathematical derivation based on standard Backpropagation calculus.