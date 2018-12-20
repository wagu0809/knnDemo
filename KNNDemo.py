import numpy as np


def str2float(s):
    isNegtive = s[0]
    s = s.strip("-")
    parts = s.split(",")
    if len(parts) == 2 and parts[1] is not "":
        fvalue = int(parts[0]) + (int(parts[1]))/(10**len(parts[1]))
        if isNegtive == "-":
            fvalue = -fvalue
    else:
        fvalue = int(s)
        if isNegtive == "-":
            fvalue = -fvalue

    return fvalue


def loaddata(eval_data_group, full_data, N):
    eval_data_set = []
    train_data_set = []
    # eval_data_group的取值为0到9, N为10 10-fold cross-validation
    if eval_data_group < N:
        full_data_size = len(full_data)
        while eval_data_group <= full_data_size - N:
            eval_data_set.append(full_data.pop(eval_data_group))
            eval_data_group += N
        train_data_set = full_data

    return train_data_set, eval_data_set


def classifier(testDataSet, trainDataSet, K):
    # trainDataSet includes training data and labels
    # split training data and labels
    trainData = []
    labels = []
    for data in trainDataSet:
        data = data.strip("\n").split("\t")
        for i in range(len(data)):
            data[i] = str2float(data[i])
        trainData.append(data[:2])
        labels.append(data[2])
    testDataSetSize = len(testDataSet)
    correctCount = 0
    incorrectCount = 0

    for i in range(testDataSetSize):
        # split testing data and label
        testData = testDataSet[i]
        testData = testData.strip("\n").split("\t")
        for i in range(len(testData)):
            testData[i] = str2float(testData[i])
        testLabel = testData[2]
        testData = testData[:2]

        # start to calculate distance
        trainData = np.array(trainData)
        trainDataSize = trainData.shape[0]
        # print(trainDataSet)
        testMat = np.tile(testData, (trainDataSize, 1))
        diffMat = testMat - trainData  # testData 和 trainDataSet的差值
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)  # 加入参数axis=1意思是让矩阵每一个行向量求和（矩阵数据相加）
        distances = sqDistances**0.5  # 计算欧式距离
        sortDistances = distances.argsort()  # argsort返回的是排序后从小到大的index值
        classcount = {}
        # print(sortDistances)
        for i in range(K):
            label = labels[sortDistances[i]]
            classcount[label] = classcount.get(label, 0) + 1
        # print(classcount)
        sortClassCount = sorted(classcount.items(), key=lambda item: item[1], reverse=False)
        # print(sortClassCount)
        class_pred = sortClassCount[0][0]
        # print(class_pred, testLabel)
        if class_pred == testLabel:
            correctCount += 1
        else:
            incorrectCount += 1
    correctRate = correctCount/testDataSetSize

    print('correct rate is: ', correctRate)


with open("./full_data.txt", encoding="utf-8", mode="r") as f:
    init_data_set = f.readlines()
    for i in range(10):
        trainDataSet, testDataSet = loaddata(i, init_data_set, 10)
        classifier(testDataSet, trainDataSet, 5)



