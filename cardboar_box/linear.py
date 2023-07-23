import numpy as np

class linear:
    def __init__(self, y, x):
        self.y = y
        self.weight = np.random.randn(y, x)
        self.bias = np.zeros((1, x))
    
    def forward(self, inputs):
        self.inputs = inputs
        nobias = np.matmul(inputs, self.weight)
        output = nobias + self.bias
        return output
    
    def backward(self,  x, lr=1):
        dw = np.matmul(self.inputs.T, x)
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

def softmax(x, derivative=False):
    if not derivative:
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        return softmax(x) * (1 - softmax(x).sum(axis=1, keepdims=True))

def relu(x,derivative = False):
    if not derivative:
        return np.maximum(0, x)
    else:
        return (x > 0).astype(float)

ins = np.array([[0.1,1,1]])
target = np.array([0,0,1])

layerone = linear(3,5)
layertwo = linear(5,3)
for s in range(20):
    x = relu(layerone.forward(ins))
    x = softmax(layertwo.forward(x))

    error = x - target 

    layertwo.backward(error)
    y = layertwo.deri(error) *  relu(layerone.forward(ins),True)
    layerone.backward(y)

    print(error)

print(x)
