#面向复杂场景下鲁棒红外小目标检测的空间–频率协同增强

## **本文已投稿至《*The Visual Computer》*.**  

源代码和模型已发布，以促进结果的可复现性及后续研究。

##概述

SF-DETR 是一种改进的 RT‑DETR 框架，专为**红外小目标检测（IRSTD）**在高度复杂场景下的应用而设计。  
我们提出了三个新颖的模块，它们协同利用**空间域与频域的协同效应**，以增强微弱目标特征并抑制复杂的背景干扰。

|模块|文件位置|核心功能|
| ------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **SFSCBlock** | `ultralytics/nn/IRSTD_modules/SFSCBlock.py` |基于深度可分离卷积、频域门控与空间校准的骨干特征提取|
| **SRAIFI**    | `ultralytics/nn/IRSTD_modules/SRAIFI.py`    |通过谱残差注意力实现尺度内特征交互（频率异常挖掘器 + 变换器）|
| **CFARepC3**  | `ultralytics/nn/IRSTD_modules/CFARepC3.py`  |用于FPN/PAFPN融合的轻量级频率细节注入与坐标注意力机制|

整个网络架构在model/SF-DETR.yaml`中定义。
所有模块均已集成到Ultralytics的YOLO/RT‑DETR框架中。

##数据集

我们在两个公开的红外小目标数据集上进行评估：

- [DAUB](https://www.scidb.cn/en/detail?dataSetId=720626420933459968)
- [HIT‑UAV](https://www.scidb.cn/en/detail?dataSetId=90340ed7f15d4b83a32094f1cfa9b39e&version=V1)

##环境

- Python 3.10

- PyTorch 2.2.2

- CUDA 12.1

  **安装依赖项：**

```
pip install -r requirements.txt
```



##引用

如果您在研究中使用本代码或数据集，请按如下方式引用我们的论文：



```
【在复杂场景下实现鲁棒红外小目标检测的空间–频率协同增强】，《视觉计算》期刊，（已投稿）。
论文DOI号（如后续有）：[将补充]
```



##致谢

This work builds upon [Ultralytics RT‑DETR](https://github.com/ultralytics/ultralytics) and uses the publicly available DAUB and HIT‑UAV datasets. We thank the contributors of these resources.
