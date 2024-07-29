"""
Collins Zhong
2024/2/1

learning how to use numpy and basic linear regression training

"""

import numpy as np;

def classification():
    X = np.array([[1,1,1],
                [1,-1,1],
                [1,1,3],
                [1,1,0]])
    Y = np.array([[1,0,0],
                [0,1,0],
                [1,0,0],
                [0,0,1]])

    K = np.arange(12).reshape(4,3)      # an array from 0 to 12, reshaped to be a 4x3 matrix from 0 to 12

    A = np.array([[1,0,0],
                [0,1,0],
                [3,0,1]])
    # print(np.linalg.inv(A))

    # @ is matrix product (can also use A.dot(B))  while  * is elementwise product
    W_hat = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y        # TRAINING


    # new input X data:
    X_new = np.array([[1,6,8],
                    [1,0,-1],
                    [0,0,0]])
    print(X_new @ W_hat)

    Y_hat = np.argmax(X_new @ W_hat, axis=1, keepdims=True)    # axis=0 是找每一列里最大的，axis=1 是找每一行, keepdims False 的话会输出美观的行向量
    print(Y_hat)

if __name__ == "__main__":
    classification()