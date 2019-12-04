from numpy import *
import numpy
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

datingTestSet = 'C:/Python/machinelearninginaction/Ch02/datingTestSet.txt'
imgTestSet = 'C:/Python/machinelearninginaction/Ch02/digits/testDigits/0_0.txt'

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    :param inX:用于分类的输入向量
    :param dataSet:训练样本集
    :param labels:训练集标签
    :param k:最近邻居的数目
    :return:
    '''
    # shape获得数组维数的tuple,shape[0]返回tuple第一个值，即dataSet的行数
    dataSetSize = dataSet.shape[0]
    # tile函数将inX转换成dataSet一样的维数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 根据欧氏距离公式求平方
    sqDiffMat = diffMat ** 2
    # 平方后相加 （x0-x1）**2+(y0-y1)**2
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号得到inX和dataSet中每个样本的距离数组
    distances = sqDistances ** 0.5
    # 对distances从小到大排序，返回的是原数组索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        #根据原数组索引值得到对应样本标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #对标签进行计数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 将classCount转换成tuple,并根据tuple的第二个元素(也就是label出现的频率)从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回频率最高的label
    return sortedClassCount[0][0]


def file2matrix(filename):
    # 用datingTestSet.txt会遇到最后一列字符串无法转int的问题，加一个字典映射或者用datingTestSet2.txt
    schema = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    with open(filename, 'r+') as fr:
        # 获得数据总行数
        numberOfLines = len(fr.readlines())
        # 按总行数进行格式化数组
        returnMat = zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
    with open(filename, 'r+') as fr:
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            # 把解析出来的listFromLine前三个元素赋值到returnMat
            returnMat[index] = listFromLine[0:3]
            # 添加listFromLine最后一个元素到classLabelVector中
            # classLabelVector.append(int(listFromLine[-1]))
            classLabelVector.append(schema[listFromLine[-1]])
            index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    # 按列得到最小数组
    minVals = dataSet.min(0)
    # 按列得到最大数组
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # normDataSet = dataSet - tile(minVals,(m,1))
    # 根据(value-min)/(max-min)进行归一化
    normDataSet = (dataSet - tile(minVals, (m, 1))) / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    #用于分类样本比例
    hoRatio = 0.1
    #datingTestSet = 'C:/Python/machinelearninginaction/Ch02/datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(datingTestSet)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVcs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVcs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVcs:m,:],datingLabels[numTestVcs:m],3)
        print("the classifier came back with:{},the real answer is:{}".format(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is:{}".format(errorCount/numTestVcs))


def classifyPerson():
    resultList = ["not at all","in small doses","in large doses"]
    percentTats = float(input("percentage of time spent playing vodeo games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix(datingTestSet)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: {}".format(resultList[classifierResult-1]))


def img2vector(filename):
    returnVect = zeros((1,1024))
    with open(filename,"r+") as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
        return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFilePath = "C:/Python/machinelearninginaction/Ch02/digits/trainingDigits"
    testingFilePath = "C:/Python/machinelearninginaction/Ch02/digits/testDigits"
    trainingFileList = listdir(trainingFilePath)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        # 获得文件名
        fileNameStr = trainingFileList[i]
        #去掉.txt后缀
        fileStr = fileNameStr.split(".")[0]
        #获得数字类型
        classNumStr = int(fileStr.split("_")[0])
        #将数字类型添加到列表中
        hwLabels.append(classNumStr)
        #将文件中的内容转换成数组
        trainingMat[i,:] = img2vector("{}/{}".format(trainingFilePath,fileNameStr))
    testFileList = listdir(testingFilePath)
    testM = len(testFileList)
    errorCount = 0.0
    for i in range(testM):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("{}/{}".format(testingFilePath,fileNameStr))
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:{},the real answer is {}".format(classifierResult,classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is :{}".format(errorCount))
    print("\nthe total error rate is :{}".format(errorCount/float(testM)))



if __name__ == '__main__':
    # group, labels = createDataSet()
    # print(classify0([2, 0], group, labels, 3))
    # datingTestSet = 'C:/Python/machinelearninginaction/Ch02/datingTestSet.txt'
    # datingDataMat, datingLabels = file2matrix(datingTestSet)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # plt.show()
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # print(normMat)
    # print(ranges)
    # print(minVals)
    # datingClassTest()
    # classifyPerson()
    # imgTestSet = 'C:/Python/machinelearninginaction/Ch02/digits/testDigits/0_13.txt'
    # testVector = img2vector(imgTestSet)
    # print(testVector[0,0:31])
    handwritingClassTest()