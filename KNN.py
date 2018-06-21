from sklearn.datasets import load_iris
import numpy as np
from tqdm import tqdm
iris=load_iris()
x=iris.data
y=iris.target
#计算data1和data2的欧几里得距离
def Distance(data1,data2):
    distance=0
    # data1和data2长度一致，这里我们就使用data1的长度
    for x in range(len(data1)):
        distance += np.square(data1[x]-data2[x])
    return np.sqrt(distance)
def KNN(x,y,testInstance, k):
    distances= dict()
    # 为了得到测试数据的预测类别，对训练集每条数据进行迭代
    for idx,trainInstance in enumerate(x):#1. 计算测试数据与训练集中每一条数据的距离
        dist=Distance(testInstance,trainInstance)
        distances[idx]=dist
    # 2. 对距离进行升序排列
    sorted_d=sorted(distances.items(),key=lambda k:k[1])
    neighbors=[]
    # 3. 对排列结果选择前K个值
    for x in tqdm(range(k)):
        neighbors.append(sorted_d[x][0])

    classvotes=dict()
    # 4. 得到出现次数最多的类
    for x in range(len(neighbors)):
        label=y[neighbors[x]]
        if label in classvotes:
            classvotes[label]+=1
        else:
            classvotes[label]=1

    # 5. 返回测试数据的预测类别
    sortedvotes=sorted(classvotes.items(),key=lambda k:k[1],reverse=True)
    return (sortedvotes[0][0],neighbors)

testInstance = [7.2, 3.6, 5.1, 2.5]
#将k设置为1
k=1
predicted,neighbors=KNN(x,y,testInstance,k)
print('predicted:',predicted)
print('neighbors:',neighbors)
'''
运行结果
predicted: 2
neighbors: [141]
'''
#将k设置为3
k = 3
predicted, neighbors = KNN(x, y, testInstance, k)
print('predicted:',predicted)
print('neighbors:',neighbors)
'''
运行结果
predicted: 2
neighbors: [141, 139, 120]
'''
#scikit-learn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x,y)
testdata=[testInstance]
predict=knn.predict(testdata)
kneighbors=knn.kneighbors(testdata)
print('predicted:',predict)
print('neighbors',kneighbors)
'''
运行结果
predicted: [2]
neighbors (array([[0.6164414 , 0.76811457, 0.80622577]]), array([[141, 139, 120]], dtype=int64))
scikit库knn算法的运行结果与我们设计的完全一致,neigbors都是[141, 139, 120]
'''
