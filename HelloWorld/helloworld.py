import numpy as np

# sigmoid 压缩函数
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# sigmoid 导函数
def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

class Net:

    def __init__(self, sizes):
        self.layers_num = len(sizes)        # 网络总层数（包含并不存在的输入层）
        self.sizes = sizes[1:]              # 各层网络节点数（不包含并不存在的输入层）
        
        # The first layer is input layer with no weight & biases
        self.biases = [np.random.randn(y) for y in sizes[1:]]   # 2-d array：self.biases[i][j] is the bias of the (i+2) layer、entry (j+1)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1],sizes[1:])]
        '''weights is 3-d array: self.weights[i][j] is the weight vector of layer (i+2)、entry (j+1),
                                   whose dimension is the dimension output from previous layer'''
        
