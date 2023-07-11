import numpy as np

X = [[1,2,3],[1,2,3],[1,2,3]]

class Layer:
    def __init__(self,inputs,neurons):
        self.weights = 0.10 * np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))
    def foward(self,batch):
        self.output = np.dot(batch,self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

Layer1 = Layer(3,4)
Layer2 = Layer(4,3)

Layer1.foward(X)
Layer2.foward(Layer1.output)
print(Layer2.output)