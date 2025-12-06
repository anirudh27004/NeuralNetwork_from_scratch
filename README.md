# MNIST Neural Network from Scratch (NumPy Only)

## 📌 Overview

This project is a complete implementation of a Deep Neural Network built entirely from scratch using **Python** and **NumPy**.

Unlike standard projects that rely on high-level frameworks like TensorFlow or PyTorch, this implementation manually handles:

* Matrix Calculus & Linear Algebra
* Forward & Backward Propagation
* Gradient Descent Optimization

**Final Accuracy:** ~85.6% on MNIST Test Set.

## 🧠 The Architecture

The network consists of an input layer, two hidden layers (using a mix of **ReLU** and **Sigmoid** activations), and an output layer using **Softmax**.

| Layer | Nodes | Activation | 
| :--- | :--- | :--- | 
| **Input** | 784 | N/A | 
| **Hidden 1** | 128 | ReLU |
| **Hidden 2** | 32 | Sigmoid |
| **Output** | 10 | Softmax | 

## 🔧 Key Implementation Details

### 1. Vectorized Operations

To ensure efficient training on the CPU, all operations are vectorized using NumPy broadcasting. This replaces slow Python loops with highly optimized C-level matrix operations.

### 2. Hybrid Activation & Backpropagation

I implemented a mixed architecture to handle non-linearity:
* **ReLU (Layer 1):** Used to prevent the vanishing gradient problem in the earlier layers.
* **Sigmoid (Layer 2):** Used to squash features into a bounded range before the final classification.
* **Derivatives:** Manually derived the gradients for both functions ($g'(z) = 1$ for $z>0$ and $g'(z) = a(1-a)$) to enable correct backpropagation.

### 3. Stability Mechanics

* **Softmax:** Implemented with numerical stability shifts (subtracting max) to prevent exponential overflow.

## 📉 Results

Training over **700 epochs** with a learning rate of **0.1**: achieved an accuracy of 85.6%



## How to Run

1. Clone the repo:

2. Install the dependencies

3. Run the notebook
