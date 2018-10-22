#使用jupyter-notebook绘制样本分布图像、生成样本数据集
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
np.random.seed(1)
import pandas as pd
dim = 100
num_classes = 3
img = Image.open('./data_img.jpg').resize((dim,dim ))
img = np.round(np.array(img)/255)
plt.figure(figsize=(10,10))
plt.imshow(img)
data_map = img*np.arange(1,num_classes+1)
data_map = np.sum(data_map,axis = -1)
plt.figure(figsize = (10,10)
plt.imshow(data_map)
mask = (np.random.random((data_map.shape))<0.05)
data_map = mask*data_map
plt.figure(figsize=(10,10))
plt.imshow(data_map)

f1 = []
f2 = []
targets = []
for i in range(1,int(data_map.max())+1):
	f1_tmp,f2_tmp = np.where(data_map == i)
	f1.extend((f1_tmp/dim).tolist())
	f2.extend((f2_tmp/dim).tolist())
	targets.extend([i]*f1_tmp.size)

data_df = pd.DataFrame({
	'F1':f1,
	'F2':f2,
	'Targets':targets
})
data_df
data_df.to_csv('data.csv')
	
