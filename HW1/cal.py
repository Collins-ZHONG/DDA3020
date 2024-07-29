import numpy as np
import cvxpy as cp

X=np.array([[1,1,1,0,0,0],
            [0,0,0,1,1,1]])
Y=np.array([[-1, -1, -2, 1, 1, 2],
            [-1, -2, -1, 1, 2, 1]])
W=cp.Variable([2,2])

# print(X[:,0])
result = np.array([0,0]).reshape(2,)
for i in range(0,6):
    result += Y[:,i] - W @ X[:,i]
obj = cp.Minimize(result)
prob = cp.Problem(obj)

prob.solve()
print(W.value)
