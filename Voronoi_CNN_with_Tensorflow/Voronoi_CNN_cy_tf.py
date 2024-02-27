# Voronoi-CNN-NOAA.py
# 2021 Kai Fukami (UCLA, kfukami1@g.ucla.edu)

## Voronoi CNN for NOAA SST data.
## Authors:
# Kai Fukami (UCLA), Romit Maulik (Argonne National Lab.), Nesar Ramachandra (Argonne National Lab.), Koji Fukagata (Keio University), Kunihiko Taira (UCLA)

## We provide no guarantees for this code_cy.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
# Ref: K. Fukami, R. Maulik, N. Ramachandra, K. Fukagata, and K. Taira,
#     "Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning,"
#     in Review, 2021
#
# The code_cy is written for educational clarity and not for speed.
# -- version 1: Mar 13, 2021

from tensorflow.python.keras.layers import Input,Add,Dense,Conv2D,Layer
from tensorflow.python.keras import Model,models
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
import os
import pickle



'''#误差计算：让模型在 cpu运行
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
'''

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(
        allow_growth=True,      #设置最小的GPU显存使用量，动态申请显存:（建议）https://blog.csdn.net/weixin_39875161/article/details/89979442
        visible_device_list="0"
    )
)
session = tf.compat.v1.Session(config=config)#config(可选)：辅助配置Session对象所需的参数(限制CPU或GPU使用数目，设置优化参数以及设置日志选项等)。
set_session(session)

# 读取数据
datasetSerial = np.arange(90000,100000,2) #5000snapshots

X = np.zeros((len(datasetSerial)*2,112,192,2))
y_1 = np.zeros((len(datasetSerial)*2,112,192,1))


# Data can be downloaded from https://drive.google.com/drive/folders/1K7upSyHAIVtsyNAqe6P8TY1nS5WpxJ2c?usp=sharing

df = pd.read_csv('/root/autodl-nas/cylinder_xx.csv',header=None,delim_whitespace=False)
dataset = df.values
x = dataset[:,:]
x_ref = x[7:119,0:192]
print(x.shape)
df = pd.read_csv('/root/autodl-nas/cylinder_yy.csv',header=None,delim_whitespace=False)
dataset = df.values
y = dataset[:,:]
y_ref = y[7:119,0:192]


omg_box = []
filename="/root/autodl-nas/Cy_Taira.pickle" 
with open(filename, 'rb') as f:
    obj = pickle.load(f)
    omg_box=obj
print(omg_box.shape)
    
    

for t in tqdm(range(len(datasetSerial))):
    omg = omg_box[t,:,:,0]
    y_1[t,:,:,0] = omg
    
    sparse_locations1 = (np.array([[76,71], [175,69],  [138,49],                   
                [41, 56], [141,61] ,[30,41],  
                [177,40],[80,55]]))
    sparse_locations = np.zeros(sparse_locations1.shape)
    sparse_locations[:,0] = sparse_locations1[:,1]
    sparse_locations[:,1] = sparse_locations1[:,0]

    sen_num = 8
    width = 112
    height = 192

    sparse_data = np.zeros((sen_num)) 
    sparse_data[0] = (omg[71,76])
    sparse_data[1] = (omg[69,175])
    sparse_data[2] = (omg[49,138])
    sparse_data[3] = (omg[56,41])
    sparse_data[4] = (omg[61,141])
    sparse_data[5] = (omg[41,30])
    sparse_data[6] = (omg[40,177])
    sparse_data[7] = (omg[55,80])


    sparse_locations_ex = np.zeros(sparse_locations.shape)
    for i in range(sen_num):
        sparse_locations_ex[i,0] = y_ref[:,0][int(sparse_locations[i,0])]
        sparse_locations_ex[i,1] = x_ref[0,:][int(sparse_locations[i,1])]
    grid_z0 = griddata(sparse_locations_ex, sparse_data, (y_ref, x_ref), method='nearest')
    X[t,:,:,0] = grid_z0
    
    mask_img = np.zeros(grid_z0.shape)
    for i in range(sen_num):
        mask_img[int(sparse_locations[i,0]),int(sparse_locations[i,1])] = 1
    X[t,:,:,1] = mask_img

for t in tqdm(range(len(datasetSerial))):
    omg = omg_box[t,:,:,0]
    y_1[len(datasetSerial)+t,:,:,0] = omg
    
    sparse_locations1 = (np.array([[76,71], [175,69],  [138,49],                   
                    [41, 56], [141,61] ,[30,41],  
                    [177,40],[80,55], [60,41],[70,60],
                    [100,60],[120,51],[160,80],[165,50],[180,60],[30,70]      
                              ]))
    sparse_locations = np.zeros(sparse_locations1.shape)
    sparse_locations[:,0] = sparse_locations1[:,1]
    sparse_locations[:,1] = sparse_locations1[:,0]

    sen_num = 16
    width = 112
    height = 192

    sparse_data = np.zeros((sen_num)) 
    sparse_data[0] = (omg[71,76])
    sparse_data[1] = (omg[69,175])
    sparse_data[2] = (omg[49,138])
    sparse_data[3] = (omg[56,41])
    sparse_data[4] = (omg[61,141])
    sparse_data[5] = (omg[41,30])
    sparse_data[6] = (omg[40,177])
    sparse_data[7] = (omg[55,80])
    sparse_data[8] = (omg[41,60])
    sparse_data[9] = (omg[60,70])
    sparse_data[10] = (omg[60,100])
    sparse_data[11] = (omg[51,120])
    sparse_data[12] = (omg[80,160])
    sparse_data[13] = (omg[50,165])
    sparse_data[14] = (omg[60,180])
    sparse_data[15] = (omg[70,30])


    sparse_locations_ex = np.zeros(sparse_locations.shape)
    for i in range(sen_num):
        sparse_locations_ex[i,0] = y_ref[:,0][int(sparse_locations[i,0])]
        sparse_locations_ex[i,1] = x_ref[0,:][int(sparse_locations[i,1])]
    grid_z0 = griddata(sparse_locations_ex, sparse_data, (y_ref, x_ref), method='nearest')
    X[len(datasetSerial)+t,:,:,0] = grid_z0
    
    mask_img = np.zeros(grid_z0.shape)
    for i in range(sen_num):
        mask_img[int(sparse_locations[i,0]),int(sparse_locations[i,1])] = 1
    X[len(datasetSerial)+t,:,:,1] = mask_img
#np.save("/root/autodl-tmp/cy_tf_X.npy", X)
#np.save("/root/autodl-tmp/cy_tf_y_1.npy", y_1)


print("载入神经网络")
#input:(批次,180,360,2)；out:(批次,180,360,1)。data_format: An optional string from: "NHWC", "NCHW",默认为"NHWC"о
class VoronioCNN(Layer):
    def __init__(self):
        super(VoronioCNN, self).__init__()
        #模型组网
        self.cnn1 = Conv2D(48, (7,7),activation='relu', padding='same')  #from tensorflow.python.keras.layers import Conv2D
        self.cnn2 = Conv2D(48, (7,7),activation='relu', padding='same')      # 构造一个二维卷积层，它具有48个输出通道和形状为（7,7）的卷积核???
        self.cnn3 = Conv2D(48, (7,7),activation='relu', padding='same')     # padding='same' ,使输出和输入具有相同的高度和宽度
        self.cnn4 = Conv2D(48, (7,7),activation='relu', padding='same')
        self.cnn5 = Conv2D(48, (7,7),activation='relu', padding='same')
        self.cnn6 = Conv2D(48, (7,7),activation='relu', padding='same')
        self.cnn7 = Conv2D(48, (7,7),activation='relu', padding='same')
        self.cnn8 = Conv2D(1, (3,3), padding='same')       #from tensorflow.python.keras.layers import Conv2D
                                    
    #前向计算
    def call(self, inputs):
        x = self.cnn1(inputs)  #from tensorflow.python.keras.layers import Conv2D
        x = self.cnn2(x)      # 构造一个二维卷积层，它具有48个输出通道和形状为（7,7）的卷积核???
        x = self.cnn3(x)      # padding='same' ,使输出和输入具有相同的高度和宽度
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = self.cnn7(x)
        x_final = self.cnn8(x)      #from tensorflow.python.keras.layers import Conv2D
        return x_final


#  载入预训练权重,返回加载好的model
def load_dygraph_pretrain(path=None):
    r"""
    Args:
        path (str): Pre-trained parameters of the model 
    """
    if not (os.path.isdir(path) or os.path.exists(path)):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    model = models.load_model(path)
    return model

#  voronioCNN（）方法实例化 model，并设置预载入模型，返回组网model
def voronioCNN() -> VoronioCNN:
    r"""
    返回input_img,outputs，用以建立训练模型：
        model = Model(input_img, outputs)
    """
    input_img = Input(shape=(112,192,2))
    outputs = VoronioCNN()(input_img)
    return input_img,outputs

#获取返回值
input_img,outputs = voronioCNN()
#建立模型
model = Model(input_img, outputs)
# 模型训练的配置准备，准备损失函数，优化器和评价指标
model.compile(optimizer='adam', loss='mse', metrics='accuracy')         #使用model.compile()方法来配置训练方法
model.summary()         #tensorflow d 的函数





print("训练参数")
from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping
X_train, X_test, y_train, y_test = train_test_split(X, y_1, test_size=0.3, random_state=None)
model_cb=ModelCheckpoint('./Model_cy.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
cb = [model_cb, early_cb]
#history = model.fit(X_train,y_train,epochs=5000,batch_size=128,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, y_test))
history = model.fit(X_train,y_train,epochs=3,batch_size=32,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, y_test))


print("存储模型")
# 存储结果
model.save('./Model_cy_tfmodel')
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./Model_cy.csv',index=False)


