import numpy as np
# from matplotlib import pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

### definition
# iteration times
def iter_times(train_num):
    iter_lst = [1000, 10000, 50000, 100000]

    return iter_lst[train_num]
""" changing iteration """

# step size
def step_size(i, n, train_num):
    size_lst = [0.001, 0.0009, 0.0007, 0.0001, (1+(n-i)/n)/500, (1-i/n)/1000]
    
    return size_lst[train_num]
""" decreasing learing rate """

# RMSE (for drawing loss decreasing curve)
def rmse(ytrue, ypred):   # the order of 2 value is not important
    return np.sqrt(mean_squared_error(ytrue, ypred))

# a = np.array([[1],[2],[3]])
# print(np.square(a))
# print(sum(np.square(a), 0)) ## [14]

### creat dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target       # 照着tut抄的
y = y.reshape(y.shape[0],1)     # adjustment for shape format of y
X_b = np.c_[np.ones(X.shape[0]), X]

### 10 training iteration_times & step_sizes choice
train_lst = [(0,0), (0,1), (0,3), (1,0), (1,2), (1,3), (1,4), (1,5), (2,4), (2,5)]

# initalize w
w = np.random.randn(X_b.shape[1],1)

### split the data before the training so that the 10 trainings use the same data, i.e. comparable
# X_train, X_test, y_train, y_test = train_test_split(X_b, y, train_size=0.8)
data = train_test_split(X_b, y, train_size=0.8)
# print(X_train.shape, X_train.T.shape, y_train.shape, w.shape)
# print((X_train@w - y_train).shape)

def gradient_descend(data, w, time, train_lst):
    X_train, X_test, y_train, y_test = data
    # print(X_train.shape, X_train.T.shape, y_train.shape, w.shape)
    # print((X_train@w - y_train).shape)

    # location_x = []
    # location_y = []

    n = iter_times(train_lst[time][0])
    for i in range(n):
        gradient = X_train.T @ (X_train@w - y_train)
        # print(w.T)
        w = w - gradient*step_size(i, n, train_lst[time][1])

        # location_x.append(i+1)
        # location_y.append(rmse(y_test, X_test@w))
        
        # if i >=n-4:
        #     print("The {}-th training RMSE is {}".format(i+1, np.sqrt(mean_squared_error(y_train, X_train @ w))))

        ### adjust the step size

    y_pre = X_test @ w
    print("The {}-th training obtains a RMSE of training of {}".format(time+1, np.sqrt(mean_squared_error(y_train, X_train@w))))
    print("The {}-th training obtains a RMSE of testing of {}".format(time+1, np.sqrt(mean_squared_error(y_test, y_pre))))
    # plt.plot(location_x, location_y)
    # plt.show()

for time in range(0,10):
    gradient_descend(data, w, time, train_lst)