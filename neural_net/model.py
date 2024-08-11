import numpy as np

"""
Computes cross entropy loss for a batch of data
Predictions y is the output of the network. It is an Nx4 matrix.
Labels t is an Nx4 matrix of one-hot encoded ground truth.
"""
def cross_entropy_loss(y, t):
    N = len(y)
    loss = 0
    for t_row, y_row in zip(t,y):
        loss += -t_row @ np.log(y_row + 0.00001)
    return loss / N


"""
Computes softmax of z where z is an Nx4 matrix (or NxK)
"""
def softmax(z):
    out = z.copy()
    for i in range(z.shape[0]):
        m = np.max(out[i])  # to prevent overflow
        out[i] = np.exp(out[i] - m)
        out[i] = out[i] / sum(out[i])
    return out


class Model:
    def forward(self, input_x):
        raise NotImplementedError
    def backward(self, input_x, t, learning_rate):
        raise NotImplementedError


"""
Simple model for testing
z = X*W + b
y = softmax(z)
"""
class OneLayer(Model):
    def __init__(self, num_input_units, num_output_units):
        self.w = np.random.randn(num_input_units, num_output_units)
        self.b = np.zeros(num_output_units)

    """
    Forward pass where input_x is an NxD matrix. Saves state needed for backpropagation
    """
    def forward(self, input_x):
        y = softmax(input_x @ self.w + self.b)
        self.y = y  # state used for backprop
        return y

    """
    Updates weights using gradient descent.
    Gradient of cost function for this model wrt to W is: 1/N X^T * (y-t)
    wrt to b: 1/N 1^T (y-t)
    """
    def backward(self, input_x, t, learning_rate):
        N = len(input_x)
        y = self.y

        w_grad = 1/N * input_x.T @ (y-t)
        self.w = self.w - learning_rate * w_grad

        b_grad = 1/N * np.ones((1,N)) @ (y-t)
        b_grad = b_grad.reshape(-1)     # convert (1,4) -> (4)
        self.b = self.b - learning_rate * b_grad


"""
Model with one hidden layer
TODO
"""
class TwoLayer(Model):
    ...

