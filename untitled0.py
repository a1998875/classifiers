#載入資料
from sklearn.datasets import load_breast_cancer
#分類器
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


DT = load_breast_cancer()
X, Y= DT.data, DT.target
tx, testx, ty, testy = train_test_split(X, Y, test_size = 0.2)

#DecisionTree
tre = DecisionTreeClassifier(criterion="entropy")
tre.fit(tx, ty)
trep = tre.predict(testx)
tacc = accuracy_score(testy, trep)
print("DecisionTree正確率:",tacc)
#RandomForest
rfc = RandomForestClassifier(max_features='auto',n_estimators=1,min_samples_leaf=5)
rfc.fit(tx, ty)
rfcp = rfc.predict(testx)
rfcacc = accuracy_score(testy, rfcp)
print("RandomForest正確率:",rfcacc)
#K-NN
knn = KNeighborsClassifier(n_neighbors=9,weights="distance")#取最近的9個
knn.fit(tx, ty)
knnp = knn.predict(testx)
kacc = accuracy_score(testy, knnp)
print("K-NN正確率:",kacc)
#SGD
sgd =SGDClassifier(loss ='hinge',penalty ='l2',alpha = 0.0001,max_iter=1000)
sgd.fit(tx, ty)
sgdp = sgd.predict(testx)
sgacc = accuracy_score(testy, sgdp)
print("SGD正確率:",sgacc)
#MLP
mlp = MLPClassifier(activation="relu",hidden_layer_sizes=(100,))
mlp.fit(tx, ty)
mlpp = mlp.predict(testx)
mlpacc = accuracy_score(testy, mlpp)
print("MLP正確率:",mlpacc)