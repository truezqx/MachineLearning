from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute','I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #1代表侮辱性，0代表正常
    classVec = [0,1,0,1,0,1]
    return postingList,classVec


def createVocabList(dataSet):
    #新建一个set
    vocabSet = set()
    #遍历数据集
    for document in dataSet:
        #把列表转换成set去重，和vocaSet并集操作并赋值给vocabSet
        vocabSet = vocabSet | set(document)
    #返回包含所有单词并去重的列表
    return list(vocabSet)

#vocabList为所有单词去重后的列表，简称单词表
#inputSet是某个文档
def setOfWords2Vec(vocabList,inputSet):
    #创建一个单词表长度的值都为0的列表returnVec
    returnVec = [0]*len(vocabList)
    #遍历文档
    for word in inputSet:
        if word in vocabList:
            #如果word在单词表中，找到word在单词表中的索引，并将returnVec中对应位置置为1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: {} is not in my Vocabulary!".format(word))
    return returnVec

#训练函数
#trainMatrix:经过setOfWords2Vec函数处理后的文档矩阵
#trainCategory:文档类型向量
def trainNB0(trainMatrix,trainCategory):
    #计算矩阵大小即文档个数
    numTrainDocs = len(trainMatrix)
    #得到单词总数
    numWords = len(trainMatrix[0])
    #分类1的概率p(y=1)，标签中1出现的次数/总文档数，得到先验概率p(c=1)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化两个全为0的矩阵
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = p1Denom = 0.0
    #对每一个文档进行训练
    for i in range(numTrainDocs):
        #如果属于侮辱性文档
        if trainCategory[i] == 1:
            #如果单词出现就+1
            p1Num += trainMatrix[i]
            #计算该类别总词数
            p1Denom += sum(trainMatrix[i])
        else:
            #同上
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #利用numpy对每个元素除以该类别总词数，得到p(wi)
    p1Vect = p1Num/p1Denom  #p(wi|c=1)
    p0Vect = p0Num/p0Denom  #p(wi|c=0)
    return p0Vect,p1Vect,pAbusive



if __name__ == "__main__":
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        print("myVocabList:{} \n postinDoc:{}".format(myVocabList,postinDoc))
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print("trainMat:{} \n listClasses:{}".format(trainMat,listClasses))
    print(len(trainMat[0]))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print("先验概率p(c=1):",pAb)
    print("每个特征的条件概率 p(w | c=1):",p1V)
    print("每个特征的条件概率 p(w | c=0):",p0V)