from math import log
import operator
# import treePlotter
import treePlotter


def calcShannonEnt(dataSet):
    #计算数据集长度
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        #获得样本标签
        currentLabel = featVec[-1]
        #字典中不存在标签则新增一个key初始化value为0
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        #累计标签出现的次数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        #计算标签在整个数据集中出现的概率
        prob = float(labelCounts[key]) / numEntries
        #求香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['no surfacing','flippers']
    return dataSet,labels


def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVet in dataSet:
        if featVet[axis] == value:
            reducedFeatVec = featVet[:axis]
            reducedFeatVec.extend(featVet[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    #统计样本特征个数
    numFeatures = len(dataSet[0])-1
    #计算出整个数据集的熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    #遍历所有特征
    for i in range(numFeatures):
        #将所有数据集中第i个特征值全部取出来
        featList = [example[i] for example in dataSet]
        #去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        #遍历所有特征值
        for value in uniqueVals:
            #对每个特征值都划分一次数据集
            subDataSet = splitDataSet(dataSet,i,value)
            #计算划分出的数据集在源数据集中的概率
            prob = len(subDataSet)/float(len(dataSet))
            #计算新数据集的熵，并求和
            newEntropy += prob * calcShannonEnt(subDataSet)
        #和原始熵进行比较
        infoGain = baseEntropy - newEntropy
        #信息增益=熵减=信息无序度减小
        if infoGain > bestInfoGain:
            #如果增益变大(熵变小)，则把当前特征当做最优特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel



if __name__=="__main__":
    # dataSet,labels = createDataSet()
    # print(dataSet)
    # shannonEnt = calcShannonEnt(dataSet)
    # print(chooseBestFeatureToSplit(dataSet))
    # print(createTree(dataSet,labels))
    myTree = treePlotter.retrieveTree(0)
    print(myTree)
    myTree['no surfacing'][3]='maybe'
    print(myTree)
    treePlotter.createPlot(myTree)