import numpy as np

X = np.array([[2, 0, 1, -1, 1, 2, -2, -2, -2, -3],
              [0, 2, 2, 1, 0, 3, 3, -2, -3, 2],
              [1, -3, 1, 3, 1, -1, -3, 2, 1, 0],
              [-3, -3, 3, 2, -1, 1, 3, 3, -2, -1],
              [-2, -2, -2, -1, 1, -2, 2, -2, -3, -2]])

# for xi in X.T:
    # print(xi)

miu = X.mean(axis=1).reshape(5,1)
# print(miu)

ans = np.zeros((5, 5))
for xi in (X-miu).T:
    xi = xi.reshape(5,1)

    ans += xi @ xi.T

# print(ans)
# print(ans/10)

E = ans/10

a = np.linalg.eigvals(ans/10)
v = np.linalg.eigvalsh(ans/10)
b = np.linalg.eig(ans/10)
# print(a)
# print(v)
# print(b)
# print(b[1])
# print(E @ b[1].T[0].reshape(5,1))
# print(0.82311524 * b[1].T[0].reshape(5,1))

# print(np.linalg.norm(b[1].T[0]))

U = np.array([b[1].T[4], b[1].T[3]]).T
print(U)

i = 0
for xi in (X-miu).T:
    xi = xi.reshape(5,1)
    i+=1
    print(i)
    print(U.T @ xi)