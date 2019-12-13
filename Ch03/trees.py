from math import log
import operator

# import treePlotter
import treePlotter


def calcShannonEnt(dataSet):
    # 计算数据集长度
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 获得样本标签
        currentLabel = featVec[-1]
        # 字典中不存在标签则新增一个key初始化value为0
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        # 累计标签出现的次数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算标签在整个数据集中出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 求香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    #遍历数据集
    for featVet in dataSet:
        #过滤掉特征值不一样的数据集
        if featVet[axis] == value:
            #通过切片去掉子集中的当前特征值，并追加子集到retDataSet中
            reducedFeatVec = featVet[:axis]
            reducedFeatVec.extend(featVet[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    # 统计样本特征个数
    numFeatures = len(dataSet[0]) - 1
    # 计算出整个数据集的熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 将所有数据集中第i个特征值全部取出来
        featList = [example[i] for example in dataSet]
        # 去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历所有特征值
        for value in uniqueVals:
            # 对每个特征值都划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # print(subDataSet)
            # 计算取当前特征值的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 计算新数据集的熵*概率，并累加
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 和原始熵进行比较
        infoGain = baseEntropy - newEntropy
        # 信息增益=熵减=信息无序度减小
        if infoGain > bestInfoGain:
            # 如果增益变大(熵变小)，则把当前特征当做最优特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    #获得数据集所有类型
    classList = [example[-1] for example in dataSet]
    #如果数据集所有类型都相同，则返回对应类型
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]
    #如果使用完了所有特征，都不能获得唯一类型，则进行挑选返回出现次数最多的类型
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #获取最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #用最优特征创建一个字典
    myTree = {bestFeatLabel: {}}
    #删除计算过得特征类型
    del (labels[bestFeat])
    #获得最优特征的所有特征值
    featValues = [example[bestFeat] for example in dataSet]
    #利用set进行去重
    uniqueVals = set(featValues)
    #遍历特征值
    for value in uniqueVals:
        #复制一个特征标签列表
        subLabels = labels[:]
        #对数据集按最优特征进行划分，递归创建tree
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    with open(filename, "w") as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    import pickle
    with open(filename, "r") as fr:
        return pickle.load(fr)


if __name__ == "__main__":
    # dataSet,labels = createDataSet()
    # print(dataSet)
    # shannonEnt = calcShannonEnt(dataSet)
    # print(chooseBestFeatureToSplit(dataSet))
    # print(createTree(dataSet,labels))
    # myTree = treePlotter.retrieveTree(0)
    # print(myTree)
    # myTree['no surfacing'][3]='maybe'
    # print(myTree)
    # treePlotter.createPlot(myTree)
    # myDat, labels = createDataSet()
    # myTree = treePlotter.retrieveTree(0)
    # print(myTree)
    # print(labels)
    # label = classify(myTree, labels, [0, 1])
    # print(label)
    # storeFilename ="C:/Python/machinelearninginaction/Ch03/classifierStorage.txt"
    # storeTree(myTree,storeFilename)
    # print(grabTree(storeFilename))
    with open("C:/Python/machinelearninginaction/Ch03/lenses.txt","r") as fr:
        lenses = [i.strip().split("\t") for i in fr.readlines()]
        print("lenses:{}".format(lenses))
        lensesLabels = ["age","prescript","astigmatic","tearRate"]
        lensesTree = createTree(lenses,lensesLabels)
        print(lensesTree)
        treePlotter.createPlot(lensesTree)