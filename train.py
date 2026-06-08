import warnings, os
warnings.filterwarnings('ignore')

from ultralytics import RTDETR
import random
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------
# 设置随机种子
# ----------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


# ----------------------------
# 主程序
# ----------------------------
if __name__ == '__main__':

    # ⭐ 模型 YAML 列表
    model_yamls = [
        r'D:\Project\zsj\sf-detr\sf-detr\ultralytics\cfg\models\rt-detr\rtdetr-r18.yaml'
    ]

    # ⭐ 数据集列表（data.yaml 文件路径）
    datasets = [
        'D:\Project\zsj\sf-detr\sf-detr\data.yaml'
    ]

    runs_per_model = 1 # 每个模型实验轮数

    # ----------------------------
    # 外层循环：实验轮数
    # ----------------------------
    for i in range(runs_per_model):
        print(f"\n==============================")
        print(f"开始第 {i+1}/{runs_per_model} 轮实验")
        print(f"==============================\n")

        # ----------------------------
        # 中层循环：数据集
        # ----------------------------
        for dataset_path in datasets:
            dataset_name = os.path.basename(dataset_path).replace('.yaml', '')

            # ----------------------------
            # 内层循环：模型
            # ----------------------------
            for yaml_path in model_yamls:
                model_name = os.path.basename(yaml_path).replace('.yaml', '')

                print(f"\n------ 训练模型: {model_name} 数据集: {dataset_name} (第 {i+1} 次) ------\n")

                # 设置随机种子
                set_seed(0)

                # 初始化模型
                model = RTDETR(yaml_path)
                # 训练
                model.train(
                    data=dataset_path,
                    cache=False,
                    imgsz=640,
                    epochs=100,
                    # patience=15,  # 早停：连续15轮验证指标不提升就停止
                    batch=4,
                    workers=0,
                    project='runs/train',
                    name='exp'
                )

                print(f"\n------ 模型 {model_name} 数据集 {dataset_name} 第 {i+1} 次实验完成 ------\n")