#jupyter-notebook
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv('./data.csv',usecols=['F1','F2','Target‘])
data[:10]
data_arr = data.values
X = data_arr[:,:2]
y = data_arr[:,-1]
X.shape,y.shape
from sklearn.cluster import KMeans
km_clf = KMeans(3).fit(X)
km_clf.cluster_centers_*100
plt.scatter(X[:,0],X[:,1])
plt.scatter(km_clf.cluster_centers_[0,0],km_clf.cluster_centers_[0,1])
plt.scatter(km_clf.cluster_centers_[1,0],km_clf.cluster_centers_[1,1])
plt.scatter(km_clf.cluster_centers_[2,0],km_clf.cluster_centers_[2,1])
km_clf.predict(X)
lable_hash = {
	1:1,
	2:2,
	0:3
}
lable_map = np.array([3,1,2])
predict = km_clf.predict(X)
predict
_y = lable_map[predict]
_y
np.sum(y == _y)/y.size

