import numpy as np

class linear:
    def __init__(self, y, x):
        self.y = y
        self.weight = np.random.randn(y, x)
        self.bias = np.zeros((1, x))
    
    def forward(self, inputs):
        nobias = np.matmul(inputs, self.weight)
        output = nobias + self.bias
        return output
    
    def backward(self, inputs, x, lr=0.1):
        dw = np.matmul(inputs.T, x)
        db = np.sum(x, axis=0)
        self.weight -= dw * lr
        self.bias -= db * lr
    
    def deri(self, x):
        d = np.dot(x, self.weight.T)
        return d
    
    def save_weights(self, filename):
        np.savez(filename, weight=self.weight, bias=self.bias)
    
    def load_weights(self, filename):
        data = np.load(filename)
        self.weight = data["weight"]
        self.bias = data["bias"]