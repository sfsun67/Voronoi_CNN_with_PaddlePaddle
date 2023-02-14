# Voronoi-CNN-NOAA.py
# 2021 Kai Fukami (UCLA, kfukami1@g.ucla.edu)

## Voronoi CNN for NOAA SST data.
## Authors:
# Kai Fukami (UCLA), Romit Maulik (Argonne National Lab.), Nesar Ramachandra (Argonne National Lab.), Koji Fukagata (Keio University), Kunihiko Taira (UCLA)

## We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
# Ref: K. Fukami, R. Maulik, N. Ramachandra, K. Fukagata, and K. Taira,
#     "Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning,"
#     in Review, 2021
#
# The code is written for educational clarity and not for speed.
# -- version 1: Mar 13, 2021

from tensorflow.python.keras.layers import Input,Add,Dense,Conv2D,merge,Conv2DTranspose,MaxPooling2D,UpSampling2D,Flatten,Reshape,LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial import Voronoi
import math
from scipy.interpolate import griddata   #python插值(scipy.interpolate模块的griddata和Rbf)


# 制定gpu使用配置。1.如果不指定是不是tensorflow使用全部资源？
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(
        allow_growth=True,      #设置最小的GPU显存使用量，动态申请显存:（建议）https://blog.csdn.net/weixin_39875161/article/details/89979442
        visible_device_list="0"
    )
)
session = tf.compat.v1.Session(config=config)
set_session(session)

# 读取数据
import h5py
import numpy as np

f = h5py.File('pallde/Voronoi-CNN-main/sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])        #180*1 single   空间分辨率
lon = np.array(f['lon'])        #360*1 single   空间分辨率
sst = np.array(f['sst'])        #64800*1914 single
time = np.array(f['time'])      #1914*1 double

sst1 = np.nan_to_num(sst)

sen_num_kind = 5
sen_num_var = 5
sen_num_kind_list = [10, 20, 30, 50, 100]  #用于训练的传感器数量:n_{sensor，train}={10，20，30，50，100}，具有5种不同的传感器位置排列，总计25个案例。为每个快照随机提供传感器位置。
sen_num_var_list = [300, 100, 200, 1, 2]

#这里1040修改为32
X_ki = np.zeros((4*sen_num_kind*sen_num_var,len(lat[0,:]),len(lon[0,:]),2))  #论文使用训练使用了从1981年到2001年的1040张快照，测试而测试快照是从2001年到2018年拍摄的。
y_ki = np.zeros((4*sen_num_kind*sen_num_var,len(lat[0,:]),len(lon[0,:]),1))
        
sst_reshape = sst[0,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')
x_ref, y_ref = np.meshgrid(lon,lat)    #np.meshgrid函数 meshgrid函数通常使用在数据的矢量化上。它适用于生成网格型数据,可以接受两个一维数组生成两个二维矩阵,对应两个数组中所有的(x,y)对。 
xv1, yv1 =np.meshgrid(lon[0,:],lat[0,:])

print("计算 voronoi 镶嵌")
# 这个循环是否是voronoi 镶嵌？
for ki in tqdm(range(sen_num_kind)):   #tqdm是Python进度条库,可以在 Python长循环中添加一个进度提示信息。
    sen_num = sen_num_kind_list[ki]
    
    X_va = np.zeros((4*sen_num_var,len(lat[0,:]),len(lon[0,:]),2))
    y_va = np.zeros((4*sen_num_var,len(lat[0,:]),len(lon[0,:]),1))
    for va in range(sen_num_var):
        
        X_t = np.zeros((4,len(lat[0,:]),len(lon[0,:]),2))
        y_t = np.zeros((4,len(lat[0,:]),len(lon[0,:]),1))
        
        for t in tqdm(range(4)):
            y_t[t,:,:,0] = np.nan_to_num(sst[t,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F'))
            np.random.seed(sen_num_var_list[va])
            sparse_locations_lat = np.random.randint(len(lat[0,:]),size=(sen_num)) # 15 sensors
            sparse_locations_lon = np.random.randint(len(lon[0,:]),size=(sen_num)) # 15 sensors

            sparse_locations = np.zeros((sen_num,2))
            sparse_locations[:,0] = sparse_locations_lat
            sparse_locations[:,1] = sparse_locations_lon

            for s in range(sen_num):
                a = sparse_locations[s,0]
                b = sparse_locations[s,1]
                while np.isnan(sst_reshape[int(a),int(b)]) == True:
                    a = np.random.randint(len(lat[0,:]),size=(1))
                    b = np.random.randint(len(lon[0,:]),size=(1))
                    sparse_locations[s,0] = a
                    sparse_locations[s,1] = b

            sparse_data = np.zeros((sen_num))
            for s in range(sen_num):
                sparse_data[s] = (y_t[t,:,:,0][int(sparse_locations[s,0]),int(sparse_locations[s,1])])
    
            sparse_locations_ex = np.zeros(sparse_locations.shape)
            for i in range(sen_num):
                sparse_locations_ex[i,0] = lat[0,:][int(sparse_locations[i,0])]
                sparse_locations_ex[i,1] = lon[0,:][int(sparse_locations[i,1])]
            grid_z0 = griddata(sparse_locations_ex, sparse_data, (yv1, xv1), method='nearest')      #python插值(scipy.interpolate模块的griddata和Rbf)
            for j in range(len(lon[0,:])):
                for i in range(len(lat[0,:])):
                    if np.isnan(sst_reshape[i,j]) == True:
                        grid_z0[i,j] = 0
            X_t[t,:,:,0] = grid_z0
            mask_img = np.zeros(grid_z0.shape)
            for i in range(sen_num):
                mask_img[int(sparse_locations[i,0]),int(sparse_locations[i,1])] = 1
            X_t[t,:,:,1] = mask_img
        
        X_va[4*va:4*(va+1),:,:,:] = X_t
        y_va[4*va:4*(va+1),:,:,:] = y_t
    X_ki[(4*sen_num_var)*ki:(4*sen_num_var)*(ki+1),:,:,:] = X_va
    y_ki[(4*sen_num_var)*ki:(4*sen_num_var)*(ki+1),:,:,:] = y_va
#以上的计算没有用到 gpu。运算非常慢。    

print("载入神经网络")
input_img = Input(shape=(len(lat[0,:]),len(lon[0,:]),2))            #from tensorflow.python.keras.layers import Input
x = Conv2D(48, (7,7),activation='relu', padding='same')(input_img)  #from tensorflow.python.keras.layers import Conv2D
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)      # 构造一个二维卷积层，它具有48个输出通道和形状为（7,7）的卷积核???
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)      # padding='same' ,使输出和输入具有相同的高度和宽度
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x_final = Conv2D(1, (7,7), padding='same')(x)       #from tensorflow.python.keras.layers import Conv2D
model = Model(input_img, x_final)                   #from tensorflow.python.keras.models import Model
model.compile(optimizer='adam', loss='mse')         #使用model.compile()方法来配置训练方法

print("训练参数")
from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping
X_train, X_test, y_train, y_test = train_test_split(X_ki, y_ki, test_size=0.5, random_state=None)   #from sklearn.model_selection import train_test_split
model_cb=ModelCheckpoint('./Model_NOAA.hdf5', monitor='val_loss',save_best_only=True,verbose=1)  #学习时遇到了keras.callbacks.ModelCheckpoint()函数，总结一下用法：官方给出该函数的作用是以一定的频率保存keras模型或参数，通常是和model.compile()、model.fit()结合使用的，可以在训练过程中保存模型，也可以再加载出来训练一般的模型接着训练。具体的讲，可以理解为在每一个epoch训练完成后，可以根据参数指定保存一个效果最好的模型。
early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)                               #深度学习技巧之Early Stopping(早停法)
cb = [model_cb, early_cb]
#使用model.fit()方法来执行训练过程，
#history = model.fit(X_train,y_train,epochs=5000,batch_size=32,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
history = model.fit(X_train,y_train,epochs=2,batch_size=2,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, y_test))

print("存储模型")
# 存储结果
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./Model_NOAA.csv',index=False)





