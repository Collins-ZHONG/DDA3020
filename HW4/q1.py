from matplotlib import pyplot as plt

x1 = [0.78, 0.62, 0.44, 0.55, 0.61, 0.47, 0.07, 0.14, 0.48, 0.32]
x2 = [0.61, 0.08, 0.62, 0.39, 0.48, 0.09, 0.38, 0.09, 0.06, 0.01]
true = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]

# threshold = 0.5

# def aaa(data, threshold, truth, )

def confusion_matrix(data, threshold, truth):
    # TP = len(list(filter(lambda x: (x > threshold and truth[data.index(x)]==1), data)))
    # TN = len(list(filter(lambda x: (x < threshold and truth[data.index(x)]==0), data)))
    # FN = len(list(filter(lambda x: (x < threshold and truth[data.index(x)]==1), data)))
    # FP = len(list(filter(lambda x: (x > threshold and truth[data.index(x)]==0), data)))
    TP = TN = FN = FP = 0
    for i, x in enumerate(data):
        # print(x)
        if (x > threshold and truth[i] == 1):
            TP += 1
        elif (x < threshold and truth[i] == 0):
            TN += 1
        elif (x < threshold and truth[i] == 1):
            FN += 1
        elif (x > threshold and truth[i] == 0):
            FP += 1


    return (TP, TN, FN, FP)
# print(confusion_matrix(x1, threshold, true))

def calculate_FPR_FNR(conf_matrix):
    TP, TN, FN, FP = conf_matrix
    FNR = FN/(TP + FN)
    # FNR = FN / 5
    FPR = FP/(FP + TN)
    # FPR = FP / 5
    return (FPR, FNR)

def calculate_TPR_FPR(conf_matrix):
    FPR, FNR = calculate_FPR_FNR(conf_matrix)
    TPR = 1- FNR
    return (TPR, FPR)


if __name__ == "__main__":

    # t = confusion_matrix(x1, 0.1, true)
    # print("TP: {}; TN: {}; FN:{}; FP:{}".format())


    # exit()

    FNR1_list = []
    FPR1_list = []

    FNR2_list = []
    FPR2_list = []

    TPR1_list = []
    TPR2_list = []

    for threshold in range(0, 100): 
        threshold /= 100
        # FNR1, FPR1 = calculate_FPR_FNR(confusion_matrix(x1, threshold, true))
        TPR1, FPR1 = calculate_TPR_FPR(confusion_matrix(x1, threshold, true))
        # print(FNR, FPR)
        # FNR1_list.append(FNR1)
        TPR1_list.append(TPR1)
        FPR1_list.append(FPR1)

        # FNR2, FPR2 = calculate_FPR_FNR(confusion_matrix(x2, threshold, true))
        TPR2, FPR2 = calculate_TPR_FPR(confusion_matrix(x2, threshold, true))
        # FNR2_list.append(FNR2)
        TPR2_list.append(TPR2)
        FPR2_list.append(FPR2)

        # print("x: {}   y: {}".format(TPR2, FPR2))
    
    # print(FNR_list, "\n", FPR_list)

    plt.figure()
    plt.ylabel("TPR(%)")
    plt.xlabel("FPR(%)")
    plt.title("ROC of both models")
    plt.xticks([i/20 for i in range(0, 21)])
    plt.yticks([i/20 for i in range(0, 21)])
    plt.plot(FPR1_list, TPR1_list, label="Model 1")
    plt.plot(FPR2_list, TPR2_list, label="Model 2")
    plt.legend(loc="upper right")
    plt.show()