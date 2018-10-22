#jupyter-notebook
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv('./data.csv',usecols=['F1','F2','Target‘])
data.head(10)
data.tail(10)
data[40:50]
data_arr = data.values
data_arr.shape
X = data_arr[:,:2]
y = data_arr[:,-1].astype(int)
X.shape,X.dtype,y.shape,y.dtype
from sklearn.neighbors import KNeighborsClassfier
knn_clf = KNeighborsClassfier().fit(X,y)
knn_clf
_y = knn_clf.predict(X)
_y
(y == _y).sum()/y.size
knn_clf.score(X,y)
plt.scatter(X[:,0][np.where(y==1)],X[:,1][np.where(y==1)])
plt.scatter(X[:,0][np.where(y==2)],X[:,1][np.where(y==2)])
plt.scatter(X[:,0][np.where(y==3)],X[:,1][np.where(y==3)])

test=[0.4,0.55]
plt.figure(figsize=(8,8))
c1 = plt.scatter(X[:,0][np.where(y==1)],X[:,1][np.where(y==1)])
c2 = plt.scatter(X[:,0][np.where(y==2)],X[:,1][np.where(y==2)])
c3 = plt.scatter(X[:,0][np.where(y==3)],X[:,1][np.where(y==3)])
cv = plt.scatter(test[0],test[1])
plt.legend([c1,c2,c3,cv],['class_1','class_2','class_3','test'],fontsize = 12)
print('knn clf predict:',knn_clf.predict([test]).irem())
