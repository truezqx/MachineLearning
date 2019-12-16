from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性，0代表正常
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    # 新建一个set
    vocabSet = set()
    # 遍历数据集
    for document in dataSet:
        # 把列表转换成set去重，和vocaSet并集操作并赋值给vocabSet
        vocabSet = vocabSet | set(document)
    # 返回包含所有单词并去重的列表
    return list(vocabSet)


# vocabList为所有单词去重后的列表，简称单词表
# inputSet是某个文档
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个单词表长度的值都为0的列表returnVec
    returnVec = [0] * len(vocabList)
    # 遍历文档
    for word in inputSet:
        if word in vocabList:
            # 如果word在单词表中，找到word在单词表中的索引，并将returnVec中对应位置置为1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: {} is not in my Vocabulary!".format(word))
    return returnVec

#词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    # 创建一个单词表长度的值都为0的列表returnVec
    returnVec = [0] * len(vocabList)
    # 遍历文档
    for word in inputSet:
        if word in vocabList:
            # 和setOfWords2Vec函数相比，唯一不同是对应值+1，而不是置为1
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 训练函数
# trainMatrix:经过setOfWords2Vec函数处理后的文档矩阵
# trainCategory:文档类型向量
def trainNB0(trainMatrix, trainCategory):
    # 计算文档个数
    numTrainDocs = len(trainMatrix)
    # 得到单词总数
    numWords = len(trainMatrix[0])
    # 分类1的概率p(y=1)，标签中1出现的次数/总文档数，得到先验概率p(c=1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 为避免出现概率为0初始化两个全为1的矩阵，分母初始化为2
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = p1Denom = 2.0
    # 对每一个文档进行训练
    for i in range(numTrainDocs):
        # 如果属于侮辱性文档
        if trainCategory[i] == 1:
            # 如果单词出现就+1
            p1Num += trainMatrix[i]
            # 计算该类别总词数
            p1Denom += sum(trainMatrix[i])
        else:
            # 同上
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 利用numpy对每个元素除以该类别总词数，得到p(wi),取对数防止小数相乘下溢四舍五入为0
    # numpy.log默然以e为底 可以log2() log10()
    p1Vect = log(p1Num / p1Denom)  # p(wi|c=1)
    p0Vect = log(p0Num / p0Denom)  # p(wi|c=0)
    return p0Vect, p1Vect, pAbusive

#vec2Classify:测试文档
#p0Vec:单词特征p(wi|c=0)的条件概率向量
#p1Vec:单词特征p(wi|c=1)的条件概率向量
#pClass1:p(c=1)的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #由于trainNB0函数中对概率做了对数处理，概率相乘的对数变为概率对数的相加
    #贝叶斯公式为p(c=1|w)=p(w|c=1)p(w)/p(c=1)，这里约去了分母p(c=1),因为同一个实例分母相同，约去不影响比较
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','stupid','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(thisDoc)
    print(testEntry,"classified as: ",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == "__main__":
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # trainMat = []
    # for postinDoc in listOPosts:
    #     print("myVocabList:{} \n postinDoc:{}".format(myVocabList, postinDoc))
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print("trainMat:{} \n listClasses:{}".format(trainMat, listClasses))
    # print(len(trainMat[0]))
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print("先验概率p(c=1):", pAb)
    # print("每个特征的条件概率 p(w | c=1):", p1V)
    # print("每个特征的条件概率 p(w | c=0):", p0V)
    testingNB()