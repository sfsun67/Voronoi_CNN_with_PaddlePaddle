from paddle import Model,set_device,optimizer,metric,callbacks,summary,load
import paddle.vision.transforms as T
import paddle.nn as nn
from paddle.static import InputSpec
from paddle.io import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.interpolate import griddata   #python插值(scipy.interpolate模块的griddata和Rbf)
import os
from typing import Any
import pandas as pd
import pickle


device = set_device('cpu') # or 'cpu'


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
#重写结构，将 TensorFlow 的数据结构换成 paddle 的。
X = np.rollaxis(X, 3, 1)    #把X_ki的3轴（2通道）位置滚动到1轴位置（）
y_1 = np.rollaxis(y_1, 3, 1) 


print("载入神经网络")
#X:(10000, 112, 192, 2);y_1:(10000, 112, 192, 1)
#in:(none,2,180,360) ;out:(none,1,180,360)。Conv2D 输入和输出是 NCHW 或 NHWC 格式，其中 N 是 batchsize 大小，C 是通道数，H 是特征高度，W 是特征宽度。
class VoronioCNN(nn.Layer):
    def __init__(self):
        super(VoronioCNN, self).__init__()
        self.cnn = nn.Sequential(
                            nn.Conv2D(2,48, (7,7), padding="SAME"),    #in_channels输入图像的通道数, out_channels由卷积操作产生的输出的通道数, kernel_size
                            nn.ReLU(),
                            nn.Conv2D(48,48, (7,7),padding="SAME"),
                            nn.ReLU(),
                            nn.Conv2D(48,48, (7,7), padding="SAME"),
                            nn.ReLU(),
                            nn.Conv2D(48,48, (7,7), padding="SAME"),
                            nn.ReLU(),
                            nn.Conv2D(48,48, (7,7), padding="SAME"),
                            nn.ReLU(),
                            nn.Conv2D(48,48, (7,7), padding="SAME"),
                            nn.ReLU(),
                            nn.Conv2D(48,48, (7,7), padding="SAME"),
                            nn.ReLU(),
                            nn.Conv2D(48,1, (3,3), padding="SAME")     
                            )
    # 执行前向计算
    def forward(self, inputs):
        x = self.cnn(inputs)
        return x

#  载入预训练权重
def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path)):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = load(path)
    model.set_dict(param_state_dict)
    return

#  voronioCNN（）方法实例化 model，并设置预载入模型，返回组网model
def voronioCNN(pretrained: bool=False, **kwargs: Any) -> VoronioCNN:
    r"""
    Args:
        pretrained (str): Pre-trained parameters of the model 
    """
    model = VoronioCNN(**kwargs)
    if pretrained:
        load_dygraph_pretrain(model, pretrained)
    return model

net = voronioCNN()

# 可视化模型组网结构和参数
params_info = summary(net,(1, 2, 180, 360))
print(params_info)

#实例化模型Input(shape=(112,192,2))
input = InputSpec((112,192), 'float32', 'x')    #这里少了一个参数：batch_size=2；optional static batch size (integer).
label = InputSpec((112,192), 'float32', 'label')
model = Model(net, input, label)

# 模型训练的配置准备，准备损失函数，优化器和评价指标
optimizers = optimizer.Adam(epsilon=1e-07,
            parameters=model.parameters())
criterion = nn.loss.MSELoss(reduction='none')
model.prepare(optimizers, criterion, metric.Accuracy())




#数据集分割  ：验证集、测试集
# X_train(50, 180, 360, 2)  ；50是指快照。应该是为了形式化，扩充到100，再分割成50；180*360空间分辨率；  最后一位【0】：grid_z0 / 【1】：mask_img
X_train, X_test, y_train, y_test = train_test_split(X, y_1, test_size=0.5, random_state=None)   #from sklearn.model_selection import train_test_split
# paddle不支持 floult64，这里做转换
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")



#自定义数据集
class ReDataset(Dataset):
    """
    步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
    """
    def __init__(self, x, y):
        self.data_dir = (x,y)

    
    """
    步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
    """
    def __getitem__(self, index):
        # 根据索引，从列表中取出一个图像
        image = self.data_dir[0][index]
        label = self.data_dir[1][index]
        return image, label
    """
    步骤四：实现 __len__ 函数，返回数据集的样本总数
    """
    def __len__(self):
        return len(self.data_dir)
train_data = ReDataset(X_train,y_train)    #重载训练数据
eval_data = ReDataset(X_test,y_test)        #重载测试数据





print("训练参数")
model_cb = callbacks.ModelCheckpoint(save_dir='./Model_NOAA_pp')      #存储训练模型
early_cb = callbacks.EarlyStopping(
    'val_loss',
    mode='min',
    patience=100,
    verbose=1,
    min_delta=0,
    baseline=None,
    save_best_model=True)
cb = [model_cb, early_cb]
#history = model.fit(X_train,y_train,epochs=5000,batch_size=32,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
#history = model.fit(X_train,y_train,epochs=2,batch_size=2,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, y_test))
history = model.fit(train_data, eval_data, batch_size=128, epochs=5000, eval_freq=1, log_freq=100, save_dir=200, save_freq=100,verbose=1, drop_last=False, shuffle=True, num_workers=0, callbacks=cb, accumulate_grad_batches=1, num_iters=None)




