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


device = set_device('cpu') # or 'cpu'


# 读取数据
import h5py
import numpy as np
f = h5py.File('/root/data/sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])        #180*1 single   空间分辨率
lon = np.array(f['lon'])        #360*1 single   空间分辨率
sst = np.array(f['sst'])        #64800*1914 single    传感器和空间是怎么映射的？64800个位置（180*360=64800）
time = np.array(f['time'])      #1914*1 double      时间
sst1 = np.nan_to_num(sst)
sen_num_kind = 5
sen_num_var = 5
sen_num_kind_list = [10, 20, 30, 50, 100]  #用于训练的传感器数量:n_{sensor，train}={10，20，30，50，100}，具有5种不同的传感器位置排列，总计25个案例。为每个快照随机提供传感器位置。
sen_num_var_list = [300, 100, 200, 1, 2]



#这里1040修改为4
X_ki = np.zeros((1040*sen_num_kind*sen_num_var,len(lat[0,:]),len(lon[0,:]),2))  #论文使用训练使用了从1981年到2001年的1040张快照，测试而测试快照是从2001年到2018年拍摄的。
y_ki = np.zeros((1040*sen_num_kind*sen_num_var,len(lat[0,:]),len(lon[0,:]),1))
        
sst_reshape = sst[0,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')
x_ref, y_ref = np.meshgrid(lon,lat)    #np.meshgrid函数 meshgrid函数通常使用在数据的矢量化上。它适用于生成网格型数据,可以接受两个一维数组生成两个二维矩阵,对应两个数组中所有的(x,y)对。 
xv1, yv1 =np.meshgrid(lon[0,:],lat[0,:])



print("计算 voronoi 镶嵌")
for ki in tqdm(range(sen_num_kind)):   #tqdm是Python进度条库,可以在 Python长循环中添加一个进度提示信息。
    sen_num = sen_num_kind_list[ki]
    
    X_va = np.zeros((1040*sen_num_var,len(lat[0,:]),len(lon[0,:]),2))
    y_va = np.zeros((1040*sen_num_var,len(lat[0,:]),len(lon[0,:]),1))
    for va in range(sen_num_var):
        
        X_t = np.zeros((1040,len(lat[0,:]),len(lon[0,:]),2))
        y_t = np.zeros((1040,len(lat[0,:]),len(lon[0,:]),1))
        
        for t in tqdm(range(1040)):
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
            X_t[t,:,:,0] = grid_z0                                                      #shape:(180, 360)
            mask_img = np.zeros(grid_z0.shape)
            for i in range(sen_num):
                mask_img[int(sparse_locations[i,0]),int(sparse_locations[i,1])] = 1     #盲猜是在指示哪个位置有传感器
            X_t[t,:,:,1] = mask_img
        
        X_va[1040*va:1040*(va+1),:,:,:] = X_t
        y_va[1040*va:1040*(va+1),:,:,:] = y_t
    X_ki[(1040*sen_num_var)*ki:(1040*sen_num_var)*(ki+1),:,:,:] = X_va
    y_ki[(1040*sen_num_var)*ki:(1040*sen_num_var)*(ki+1),:,:,:] = y_va
#重写结构，将 TensorFlow 的数据结构换成 paddle 的。
X_ki = np.rollaxis(X_ki, 3, 1)    #把X_ki的3轴（2通道）位置滚动到1轴位置（）
y_ki = np.rollaxis(y_ki, 3, 1) 


print("载入神经网络")
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
                            nn.Conv2D(48,1, (7,7), padding="SAME")     
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

#实例化模型
input = InputSpec((len(lat[0,:]),len(lon[0,:])), 'float32', 'x')    #这里少了一个参数：batch_size=2；optional static batch size (integer).
label = InputSpec((len(lat[0,:]),len(lon[0,:])), 'float32', 'label')
model = Model(net, input, label)

# 模型训练的配置准备，准备损失函数，优化器和评价指标
optim = optimizer.Adam(learning_rate=0.1,epsilon=1e-07,
                parameters=model.parameters())     #参数是默认的，paddleepsilon (float，可选) - 保持数值稳定性的短浮点类型值，默认值为 1e-08。另一个epsilon=1e-07
model.prepare(optim,
            nn.loss.MSELoss(),
            metric.Accuracy())




#数据集分割  ：验证集、测试集
# X_train(50, 180, 360, 2)  ；50是指快照。应该是为了形式化，扩充到100，再分割成50；180*360空间分辨率；  最后一位【0】：grid_z0 / 【1】：mask_img
X_train, X_test, y_train, y_test = train_test_split(X_ki, y_ki, test_size=0.5, random_state=None)   #from sklearn.model_selection import train_test_split
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
history = model.fit(train_data, eval_data, batch_size=32, epochs=5000, eval_freq=1, log_freq=10, save_dir=20, save_freq=10,verbose=1, drop_last=False, shuffle=True, num_workers=0, callbacks=cb, accumulate_grad_batches=1, num_iters=None)




