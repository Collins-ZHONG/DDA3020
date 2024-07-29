import numpy as np

# # c = np.array([[1],[2]])
# a = np.array([[0],[1]])
# def euclidean_distance(x, y):
#     return np.sqrt(np.sum(np.square(x-y)))

# # print(euclidean_distance(a,c))

# # print(np.linalg.norm(a-c))

# x = np.array([[1,2],
#               [3,4],
#               [1,3],
#               [4,2]])
# c = np.array([[0,1],
#               [3,2]])

# # print(x-a)
# # print(np.linalg.norm(x-a))
# lst = []
# for i in range(2):
#     lst.append(np.linalg.norm((x-c)[i]))

# print(np.min(np.array(lst)))

# for i in x:
#     print(c-i)

# a = [1,2,3,3,3]
# print(a.count(2))


# a = np.array([[1,2]])
# a = np.array([1,2])
# c = np.array([[0,0],
#               [1,1],
#               [2,2]])
# # c = np.array([[2,2]])
# print(a-c)
# print(a.T-c.T)
# print(np.linalg.norm(a - c, axis=0))

# print(np.argmin(np.linalg.norm(a - c, axis=0)))

# a = [1,1,1,1,1]
# b = [1,2,3,4,5]
# print(a/b)
### error


# a = [np.array([[0], [0]])] *3
# a = [np.array([[0],[0]]), np.array([[0],[0]]), np.array([[0],[0]])]
# a = []
# for _ in range(3):
#     a.append(np.zeros([2,1]))
# print(a)
# a[2] += np.array([[2], [1]])
# print(a)

# # for i in a:
# #     print(id(i))

# a[2] /= 2
# print(a)

# print(np.zeros(2).reshape(1,2))

# x = np.array([1,1,2,2,3,3,4,4,5,5,6,6]).reshape(6,2)
# # print(x)
# c = np.array([[0,0],
#               [1,1],
#               [2,2]])
# distances = []

# for c_ in c:
#     # print(x-c_) 
#     # 符合预期，每行为坐标距离（difference in x, difference in y）
#     # print((x-c_).shape)       # (N, D)
#     distance = np.linalg.norm(x-c_, axis=1)
#     # print(distance)
#     # print(type(distance))   # distance is a numpy.ndarray
#     # print(distance.shape)   # with shape (N,)

#     # distance 是每个数据点到 center 的距离

#     distances.append(distance)

# # print(distances)  # a list containing K arrays
# distances = np.array(distances)
# print(distances)

# print(np.argmin(distances, axis=1))



# t = np.array([1,2,3,4,5,6]).reshape(6,)
# print(t.shape)

# # print([i for i in t])
# a = list(i for i in t)
# print(a.count(1))



x = np.array([1,1,2,2,3,3,4,4,5,5,6,6]).reshape(6,2)
# miu = np.array([1,1])
# # print(x - miu)
# sigma = np.array([a.reshape(2,1) @ a.reshape(1,2) for a in (x-miu)])
# # print(np.array([a.reshape(2,1) @ a.reshape(1,2) for a in (x-miu)]))       # 符合预期
# # print(np.array([(a @ a.T) for a in (x-miu)]))

# print(np.sum(sigma, axis=0))
index = np.array([1,1,1,2,2,1])

print(x[index==2])