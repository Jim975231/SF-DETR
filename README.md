# Spatial–Frequency Synergistic Enhancement for Robust Infrared Small Target Detection in Cluttered Scenes

**Our paper has been submitted to The Visual Computer. The code is uploaded first.**

```
SF-DETR is an improved RT-DETR framework designed for infrared small target detection (IRSTD).

The proposed method introduces:

- SRAIFI (Spatial Receptive-field Adaptive Infrared Feature Interaction)
- SFSC Block (Spatial-Frequency Synergistic Convolution Block)
- CFARepC3 (Cross-scale Feature Aggregation RepC3)

to enhance weak target representation and suppress complex background interference.


The overall network architecture of the model is specified in model/SF-DETR.yaml. 
The implementation details of the three core modules—SRAIM, Light-FDE, and CFARepC3—can be found in the following files:
ultralytics/nn/IRSTD_modules/SRAIFI.py
ultralytics/nn/IRSTD_modules/SFSCBlock.py
ultralytics/nn/IRSTD_modules/CFARepC3.py


```

---

## Datasets

```
The DAUB dataset can be accessed at:

[DAUB Dataset](https://www.scidb.cn/en/detail?dataSetId=720626420933459968)

The HIT-UAV dataset is available at:

[HIT-UAV Dataset](https://www.scidb.cn/en/detail?dataSetId=90340ed7f15d4b83a32094f1cfa9b39e&version=V1)
```

---

## Environment

```bash
python 3.10
torch 2.1.0
cuda 12.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Citation

If you use this code or data, please cite the following:
[Spatial–Frequency Synergistic Enhancement for Robust Infrared Small Target Detection in Cluttered Scenes], submitted to The Visual Computer.
