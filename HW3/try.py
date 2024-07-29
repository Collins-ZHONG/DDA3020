import numpy as np

# # Create a sample matrix
# matrix = np.array([[1, -2, 3],
#                     [-4, 5, -6],
#                     [7, -8, 9]])

# # Apply ReLU function element-wise
# relu_matrix = np.maximum(matrix, 0)

# print("Original Matrix:")
# print(matrix)

# print("\nMatrix after applying ReLU:")
# print(relu_matrix)

# A = np.array([[1,0,1,1,2,5,2,7,0],
#               [7,2,8,2,1,0,5,3,8],
#               [3,1,6,8,8,3,5,2,1]])
# # print(np.max(A,axis=1)) # max function return an 1*n array
# tmp = np.max(A, axis=1).reshape(3,1)
# # print(tmp)
# A -= tmp  # Property that it doesn't matter for all softmax to minus the same number

# A = np.exp(A)
# total = np.sum(A, axis=1).reshape(A.shape[0], 1)   ## sum for every lines
# A = A / total

# print(A)

# print(np.sum(A, axis=1))


# # Create two sample matrices
# matrix1 = np.array([[1, 2, 3],
#                      [4, 5, 6]])

# matrix2 = np.array([[2, 2, 3],
#                      [3, 5, 7]])

# # Compute Mean Squared Error (MSE) between the two matrices
# mse = np.mean((matrix1 - matrix2) ** 2)

# print("Matrix 1:")
# print(matrix1)

# print("\nMatrix 2:")
# print(matrix2)

# print("\nMean Squared Error (MSE) between the two matrices:", mse)


# import numpy as np

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# # Example input array
# x = np.array([[-1, 2, -3],
#               [4, -5, 6]])

# # Compute the derivative of the ReLU function
# relu_deriv = relu_derivative(x)

# print("Input Array:")
# print(x)

# print("\nDerivative of ReLU Function:")
# print(relu_deriv)


# import numpy as np

# X = np.array([[1,2],
#               [3,4]])
# W = np.array([[1,1], [2,2]])
# b = np.array([1,2]).reshape(2,1)
# print(X* W)
# print(W@X)
# print(W@X+b)

# import numpy as np

# # 创建包含多个 NumPy 数组的列表
# a = [np.array([[1, 2], [3, 4]]),
#      np.array([[5, 6], [7, 8]])]

# # 在轴 0 上连接这些数组
# result = np.concatenate(a, axis=0)

# print("连接后的数组：")
# print(result)


# import numpy as np

# # 创建一个示例数组
# arr = np.array([[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9]])

# # 沿着指定轴查找最大值的索引
# max_index = np.argmax(arr)
# print("最大值的索引：", max_index)

# # 沿着指定轴查找每行最大值的索引
# max_indices_row = np.argmax(arr, axis=1)
# print("每行最大值的索引：", max_indices_row.reshape(max_indices_row.shape[0],1))

# # 沿着指定轴查找每列最大值的索引
# max_indices_col = np.argmax(arr, axis=0)
# print("每列最大值的索引：", max_indices_col.reshape(max_indices_col.shape[0],1))

# print(np.random.randn(2, 2, 16) * np.sqrt(2/68))
# print(np.zeros((1, 1, 1, 16)))


# print(np.sum(np.array([[1,1,1,1],
#                        [1,1,1,1],
#                        [1,1,1,1]])))

print(10e-9)