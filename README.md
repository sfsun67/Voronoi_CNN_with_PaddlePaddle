# 基于Voronoi嵌入辅助深度学习的稀疏传感器全局场重建

本文介绍 AI for Science 共创计划（第二期），科研论文复现。结合各类 Al 网络模型在CFD/CAE等领域的应用，探索基于飞桨的复现工作。

本文是加州大学洛杉矶分校 Kai Fukami 等人于2021年发表在 ***[Nature* *Machine* *Intelligence](http://www.baidu.com/link?url=A51xmQ8vcHZoo6ieQC2gCZTR00iLwsGizbk5aRY1BIAm_oKVzH8fC4WZ530VbzwI)*** 上的文章。

本项目基于 PaddlePaddle 框架复现使用 TensorFlow 框架的稀疏传感器全局场重建技术。

# **一、论文简介**

在复杂的物理系统中，利用有限的传感器信息重建空间场是一项挑战。本文讨论了使用有限数量的传感器准确、可靠地理解随时间推移而变化的复杂场的问题。当传感器以看似随机或无组织的方式稀疏放置，以及当传感器可以处于运动状态并随着时间的推移变为在线或离线时，这个问题就变得更具挑战性。本研究提出的解决方案是一种数据驱动的空间场恢复技术，该技术使用基于结构化网格的深度学习方法对任意位置的传感器进行任意放置。

对机器学习的直接使用对于全局场重建来说并不可靠，并且无法适应任意数量的传感器。为了解决这个问题，本文使用Voronoi细分从传感器位置获得结构化网格表示，这使得卷积神经网络的使用能够在普通数据上进行处理。

所提出的方法的核心特征之一是它与为图像处理而建立的结构化传感器数据的基于深度学习的超分辨率重建技术兼容。针对非定常尾流、地球物理数据和三维湍流，演示了所提出的重建技术。当前的框架能够处理任意数量的移动传感器，这克服了现有重建方法的主要限制。所提出的技术为实际使用神经网络进行实时全局场估计开辟了一条新途径。

### **1.1 论文信息**

- 论文地址：[https://www.nature.com/articles/s42256-021-00402-2](https://www.nature.com/articles/s42256-021-00402-2)
- 参考项目：[https://github.com/kfukami/Voronoi-Cnn](https://github.com/kfukami/Voronoi-Cnn)

### **1.2 实现方法**

提出了一种基于Voronoi嵌入的深度学习的稀疏传感器全局场重建的方法，用于从二维圆柱尾流中离散传感器位置恢复全局数据。

下图描述了数据驱动的空间场恢复技术的具体应用，该技术是使用Voronoi嵌入重建二维圆柱尾流。Voronoi 细分是一种数学方法，它根据点或传感器的距离将空间划分为多个区域。在这种情况下，Voronoi 图像由 8 个传感器构成，这些传感器稀疏地位于尾流周围。 然后，Voronoi 图像被输入卷积神经网络 (CNN) 中。在这种情况下，图像的网格带有传感器（由蓝色圆圈表示），其值为 1，而所有其他区域的值均为 0。这样可以确保CNN仅处理传感器周围的区域，从而降低了重建的计算成本。

![](https://ai-studio-static-online.cdn.bcebos.com/f38000e7bfc9429a83363e8ea465c4042742c3f8a2e2476784b379b3b6e65fb3)

### **1.3 预测结果与解释**

本文所提出的技术与基于深度学习的超分辨率重建技术兼容，这些技术用于为图像处理而建立的结构化传感器数据。这意味着该技术可用于从低分辨率传感器数据中重建高分辨率图像。

该技术已针对非定常尾流、地球物理数据和三维湍流，演示了所提出的重建技术，并且能够处理任意数量的移动传感器，这克服了现有重建方法的主要局限性。 总体而言，所提出的技术为实际使用神经网络从稀疏和无组织的传感器数据进行实时全球场估算提供了新的途径。

以下示例使用Voronoi细分对传感器数据进行数据驱动空间场恢复技术的演示。使用输入的 Voronoi 图像和输入掩码图像保存传感器位置来重建涡度场。通过改变训练快照的数量来测试重建能力，训练快照本质上是用于训练神经网络的数据量。

**二维圆柱尾迹（非定常尾流）：**

结果表明，本文所提出的深度学习技术可以准确、详细地重建近尾和远尾的涡旋和剪切层。由于传感器数量少，nsensor = 8 的涡度场显示了一些低级重建误差。当 nsensor 加倍到 16 时，通过精确恢复全局流场，重建中的误差将减少一半。此外，在 nsensor = 16 时，只需使用 50 个训练快照即可实现合理的数据恢复。

![](https://ai-studio-static-online.cdn.bcebos.com/fe1806a966d74e50a6ddf977ddde6c61f860c9d29c5b4f16b3594da5dae09774)

**NOAA海面温度（地球物理数据）：**

图 3 比较了 nsensor = 30 时 NOAA 海面温度的空间场恢复，而图 4 显示了基于Voronoi细分对不同数量的传感器（包括训练期间不在线的传感器）的 NOAA 海面温度的空间场恢复。

结果表明，即使传感器发生位移，该技术也能成功地重建全局场，突显了掩模图像中保存传感器位置信息的有效利用。即使不可见的传感器数量达到 200，该技术也能在 L2 误差小于 0.1 的情况下实现合理的估计。这表明，对于传感器的数量和传感器放置位置差异很大的数据集，该方法是可靠的。

![](https://ai-studio-static-online.cdn.bcebos.com/801208a107e243cc96ddb084cead8981e47d085f9cd94db1b0ec09aab92523e1)
![](https://ai-studio-static-online.cdn.bcebos.com/9a6e1ac2d5a348878de26d7e9fbe705905dae6fcd22b432686bf96ada16ab41b)

**湍流槽道流（三维湍流）：**

对于传感器数量 = {150，200，250}，图5总结了基于Voronoi辅助CNN的湍流槽道流空间数据恢复的性能。相对于场上网格点的数量，这些传感器数量分别为2.44％、3.26％和4.07％。该研究观察到，仅用200个传感器就可以精确地重建出更精细的湍流槽道流特征，这表明测量的稀疏程度非常高。尽管不可见位置（即传感器数量相同但位置不同）的误差水平高于经过训练的传感器放置的误差水平，但可以获得相似的趋势。值得注意的是，传感器数量 = {150，250} 的结果显示了对不可见的传感器数量和位置的合理重建。结果表明，本方法是通过稀疏传感器测量对复杂流场进行全局重构的有力工具。

![](https://ai-studio-static-online.cdn.bcebos.com/bfe690fec66a422fae729faf8eb676c24d9ef6551c444c9193b9fe8ccf45789d)

# **二、复现精度**

### **2.1 验收标准**

原论文中并没有模型的评价指标，因此使用原论文作者给出的 TensorFlow 模型作为复现基准。下面展示Voronoi嵌入辅助的湍流槽道流数据全局场重建模型， Model_2Dxysec 程序， TensorFlow 转换成 PaddlePaddle 对齐过程中的具体指标。Model_noaa 程序和Model_cy 程序在涉及深度学习的架构上完全相同。

复现的验收标准如下：

| 步骤 | 阈值 | 额外说明 |
| --- | --- | --- |
| 前向对齐 | e-6 | 前向对齐验证模型的网络结构代码转换、权重转换、模型组网的正确性。一般小于1e-6的话，可以认为前向没有问题。 |
| 损失函数对齐 | e-5 | 计算loss的时候，建议设置model.eval()，避免模型中随机量的问题。 |
| 反向对齐 |  | 反向对齐验证两个代码的训练超参数全部一致。经过2轮以上，loss均可以对齐，则基本可以认为反向对齐。 |
| 训练对齐 | 0.01 | 由于是精度对齐，Model_2Dxysec 程序在全量数据训练下的 loss 精度 diff 在 1% 以内时，可以认为对齐，因此将diff_threshold 参数修改为了0.01 。 |

### **2.2 指标实现情况**

复现的实现指标如下：

```
Model_2Dxysec 程序：
forward_diff：4.3013125150537235e-07
loss_diff：1.9857959614455467e-06
bp_align_diff：
train_align_diff：0.007732505130767842

Model_noaa 程序：
forward_diff：4.3013125150537235e-07
loss_diff：1.9857959614455467e-06
bp_align_diff：loss_0:0.0    loss_1:2.6332460038247518e-05    loss_2:1.3369167390919756e-05
```

其中，各项指标均在验收标准范围内。Model_noaa 程序和Model_cy 程序在涉及深度学习的架构上完全相同。

### **2.3 复现地址**

论文复现地址：

AI Studio: [https://aistudio.baidu.com/aistudio/projectdetail/5807904](https://aistudio.baidu.com/aistudio/projectdetail/5807904)

Github: [https://github.com/sfsun67/Voronoi_CNN_with_PaddlePaddle](https://github.com/sfsun67/Voronoi_CNN_with_PaddlePaddle)

# **三、数据集**

- 数据源（Google Drive）：
    - Example 1 (two-dimensional cylinder wake): https: [//drive.google.com/drive/folders/1K7upSyHAIVtsyNAqe6P8TY1nS5WpxJ2c?usp=sharing](notion://drive.google.com/drive/folders/1K7upSyHAIVtsyNAqe6P8TY1nS5WpxJ2c?usp=sharing)),
    - Example 2 (NOAA sea surface temperature): [https://drive.google.com/drive/folders/1pVW4epkeHkT2](https://drive.google.com/drive/folders/1pVW4epkeHkT2) WHZB7Dym5IURcfOP4cXu?usp=sharing)
    - Example 3 (turbulent channel flow): [https://drive.google.c](https://drive.google.c/) om/drive/folders/1xIY_jIu-hNcRY-TTf4oYX1Xg4_fx8ZvD?usp=sharing).
- 说明：
    - NOAA 全局场重建中，Voronoi 嵌入计算过程耗时较多。可以使用代码将Voronoi 嵌入计算结果保存下来，方便调整训练模型。
        
        ```python
        np.save("./X.npy", X)
        np.save("./Y.npy", Y1)
        ```
        

# **四、环境依赖**

- 硬件：
    - GPU、CPU。
    - Model_cy ，Model_noaa，对机器内存要求较高。建议机器内存 ≥ 80G，显存 ≥ 80G。
- 框架：
    - PaddlePaddle >= 2.4.0。
    - Python:  3.7.4

# **五、快速开始**

**step1：克隆本项目**

搜索Voronoi_CNN_with_PaddlePaddle，选择对应的版本，Fork。

![](https://ai-studio-static-online.cdn.bcebos.com/3ea3e052b5e943dda5c8adcfce681826c367e1c429c64a1d83c9e7563f7a8ada)
**step2：开始训练**

选择进入终端：

![](https://ai-studio-static-online.cdn.bcebos.com/8f8fca1fc82741d1b56aad66353fbd74aa1c498727ff4499bd6c46e6e5b91f31)

单卡训练：

```
**训练** Model_2Dxysec：
		python /home/aistudio/Voronoi_CNN_with_PaddlePaddle/Model_2Dxysec/Voronoi_CNN_ch2Dxysec_pp.py

**训练** Model_noaa：
		python /home/aistudio/Voronoi_CNN_with_PaddlePaddle/Model_noaa/Voronoi_CNN_NOAA_pp.py

**训练** Model_cy：
		python /home/aistudio/Voronoi_CNN_with_PaddlePaddle/Model_cy/Voronoi_CNN_cy_pp.py

```

训练结果保存在 Model_2Dxysec_pp 、Model_cy_pp或 Model_NOAA_pp 文件中，用于存放训练日志：

```
.
├── Model_2Dxysec_pp
│   ├── XXX.pdopt
│   ├── XXX.pdparams
│   ├── ...
├── Model_cy_pp
│   ├── XXX.pdopt
│   ├── XXX.pdparams
│   ├── ...
├── Model_NOAA_pp
│   ├── XXX.pdopt
│   ├── XXX.pdparams
│   ├── ...
└────── 

```

部分训练日志如下所示

```
Epoch 1153/5000
step 1/1 [==============================] - loss: 0.1624 - acc: 0.1071 - 35ms/step
save checkpoint at /home/aistudio/Model_cy_pp/1152
Eval begin...
step 1/1 [==============================] - loss: 0.2205 - acc: 0.0402 - 13ms/step

Epoch 1154/5000
step 1/1 [==============================] - loss: 0.1613 - acc: 0.0938 - 35ms/step
save checkpoint at /home/aistudio/Model_cy_pp/1153
Eval begin...
step 1/1 [==============================] - loss: 0.1715 - acc: 0.0536 - 13ms/step
```

# **六、代码结构与参数说明**

### **6.1 代码结构**

```
.
├─ Voronoi_CNN_with_PaddlePaddle
│   ├── Model_2Dxysec
│           ├─ Voronoi_CNN_ch2Dxysec_pp.py
│           └─ Voronoi_CNN_ch2Dxysec_tf.py
│   ├── Model_noaa
│           ├─ Voronoi_CNN_NOAA_pp.py
│           └─ Voronoi_CNN_NOAA_tf.py
│   └─  Model_cy
│           ├─ Voronoi_CNN_cy_pp.py
│           └─ Voronoi_CNN_cy_tf.py
├─ data
│   ├── ch_2Dxysec.pickle
│   ├── Cy_Taira.pickle
│   ├── cylinder_xx.csv
│   ├── cylinder_yy.csv
│   ├── record_x.csv
│   ├── record_y.csv
│   └── sst_weekly.mat
├─ main.ipynb
├─ Model_2Dxysec_pp
│   ├── XXX.pdopt
│   ├── XXX.pdparams
│   ├── ...
├─ Model_cy_pp
│   ├── XXX.pdopt
│   ├── XXX.pdparams
│   ├── ...
├─ Model_NOAA_pp
│   ├── XXX.pdopt
│   ├── XXX.pdparams
│   ├── ...
└────────────────────

```

### **6.2 参数说明**

以 Model_2Dxysec 为例，其他模型相同，包括以下内容：

| 参数 | 推荐值 | 额外说明 |
| --- | --- | --- |
| batch_size | 256 | 批量 |
| learning_rate | 0.001 | 默认 |
| epochs | 5000 |  |
| optimizer | Adam | epsilon=1e-07，与 TensorFlow 模型一致 |
| criterion | MSELoss | 损失函数 |
| EarlyStopping | callbacks.EarlyStopping(
'val_loss',
mode='min',
patience=100,
verbose=1,
min_delta=0,
baseline=None,
save_best_model=True) | 早停法 |

# **七、模型信息**

| 信息 | 说明 |
| --- | --- |
| 发布者 | SSHAFER |
| 时间 | 2023.03 |
| 框架版本 | Paddle2.4.0 |
| 应用场景 | 稀疏传感器全局场重建 |
| 支持硬件 | GPU、CPU |
| 原论文 | Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning |

# **八、复现心得**

1. 本次复现主要难度体现在 TensorFlow 转换成 PaddlePaddle 过程中。转换的过程中，不仅仅只是改写模型组网代码，还需要对转换后的模型进行精度验证。这一步可以参考飞桨官方给出的 [论文复现指南-CV方向](https://github.com/PaddlePaddle/models/blob/release/2.2/docs/lwfx/ArticleReproduction_CV.md) 但里面很多代码需要自己重写。后面，我会另外写一篇将 TensorFlow 转换成 PaddlePaddle 模型的思路。
2. 如果是初次做论文复现的工作。建议至少过完一半李沐在 B 站的深度学习视频。这样在复现的过程中，进度会比较快（不会卡在很多奇怪的地方）。
3. 在 PaddlePaddle 模型组网过程中将组网方式从使用 [paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Sequential_cn.html#sequential) 组网：改为使用 [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#layer) 组网，便于后续模型对齐。
    
    ```python
    print("载入神经网络")
    class VoronioCNN(nn.Layer):
        def __init__(self):
            super(VoronioCNN, self).__init__()
            self.cnn = nn.Sequential(
                                nn.Conv2D(2,48, (7,7), padding="SAME"),   
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
    ```
    
4. 在 PaddlePaddle 模型数据载入过程中，需要使用 paddle.io.Dataset 自定义数据集（文档 - 使用指南 - 模型开发入门 - 数据集定义与加载）
    
    ```python
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
    train_data = ReDataset(X_train,y_train)     #重载训练数据
    eval_data = ReDataset(X_test,y_test)        #重载测试数据
    ```
    
5. 对齐核验过程中，需要自己准备有 GPU 的机器，因为飞桨的 AI Studio 不支持除了 PaddlePaddle 以外的其他深度学习框架。如果没有自己的机器，可以找云服务器。先装 GPU 版本的TensorFlow，然后装 CPU 版本的 PaddlePaddle，尽可能保持环境的稳定，减少配置环境花费的时间。同时，使用复现论文时，官方在 AI Studio 提供的PaddlePaddle GPU算力。这样，在自己的机器上做 TensorFlow 和 PaddlePaddle 模型的对齐工作。在 AI Studio 上验证 pp 模型的正确性。
6. 注意，前向对齐中。一开始使用小样本（n=10）前向对齐指标在e-6，无法下降，应该是遇到了两个模型卷积的 padding 方式不同。增大样本量（n=100）之后，前向对齐指标可以下降到 e-7。原理：根据中心极限定理，样本量增加，样本均值逼近正态分布，保证了数据的稳定性。
7. 注意，paddle 和 torch 中，模型推断用 modle.eval() 函数，使模型进入推断模式，只进行前向执行。在该模式下，模型会根据给定的输入直接计算输出，而不会更新模型中的权重参数。TensorFlow 中，使用model.predict() 对训练好的模型进行预测；TensorFlow 中，使用model(data, training=False) 确保它在推理模式下运行。
    
    TensorFlow 官方给出的说法是，model.predict() 这个方法适用于大批量的数据，而model(data, training=False) 这个方法没有特别说明。实验数据表明，两个方法的计算结果是一致的。
    
![](https://ai-studio-static-online.cdn.bcebos.com/96fba384daa648909f2bfacd963d8d21a8b0dda196bb4366b4a631ea34e499cd)
