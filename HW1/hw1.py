import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 忽略奇奇怪怪的报错
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


if __name__ == "__main__":

    """ Step 1 """
    ### load csv file into pandas Dataframe
    df = pd.DataFrame(pd.read_csv("C:/Users/Collins/Python/MachineLearning/HW1/house_prices.csv"))
    # print(data.columns)

    ### convert the data type
    df["Neighborhood"] = df["Neighborhood"].astype("category")

    # print(data.info())
    # print(data.describe())      # category type would be ignored
    sample_size = df.describe().loc["count","SqFt"]


    """ Step 2 """

    # # sns.pairplot(df,hue="Neighborhood")
    ### plot Price on the other 3 variables with colors by Neighborhood variable
    # sns.pairplot(df, hue="Neighborhood",x_vars=["SqFt","Bedrooms","Bathrooms"], y_vars=["Price"])
    # plt.show()
    # sns.heatmap(df.drop(["Neighborhood"], axis=1).corr(), annot=True)
    # plt.show()


    """ Step 3 """
    ### convert the category column into a one-hot numeric matrix in the dataset
    # category = OneHotEncoder(categories=ColumnTransformer(df["Neighborhood"]))
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Neighborhood'])], remainder="passthrough")   # suggested by ChatGPT
    ### a method similar to Dummy Variable for dealing with categories
    # print(category, type(category))
    print(ct)

    X = df.drop(["Price"], axis=1)
    y = df["Price"]
    X_encode = np.array(ct.fit_transform(X))
    # print(X_encode)

    # df = df.drop(["Neighborhood"],axis=1)     有了passthrough应该就不用了
    

    ### split data into train and test subset
    # 80%--> train samples ;  20%--> test samples
    train_X, test_X, train_y, test_y = train_test_split(X_encode, y, test_size=0.2, train_size=0.8)      
    # return a pure list, lst[0] is the training set, lst[1] is the test set
    print("train size: ", train_X.shape[0])
    print("test size: ", test_X.shape[0])


    """ Step 4 """
    
    # X = data[0].iloc[:, 0:-1]
    # y = data[0].loc[:,["Price"]]
    # print(X,"\n",y)
    model = LinearRegression()
    model.fit(train_X,train_y)
    
    print("intercept_:\n", model.intercept_)
    print("coef_:\n", model.coef_)
    ''' [ -6764.42605547 -17818.64080379  24583.06685926     37.02232114
    681.58966      9275.47326173]    why '''

    y_pre = model.predict(test_X)        ### test_X @ model.coef_ + model.intercept_
    print("Training RMSE: ", np.sqrt(mean_squared_error(train_y, model.predict(train_X))))
    print("Test RMSE: ", np.sqrt(mean_squared_error(test_y, y_pre)))
