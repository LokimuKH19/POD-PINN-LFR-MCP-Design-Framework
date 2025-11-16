import numpy as np
from matplotlib import pyplot as plt
import json
import pandas as pd
import scienceplots

# ==============================
# 全局颜色与字体配置
# ==============================
COLORS = {
    "train": "#71B8FF",  # 蓝色
    "test": "#FFB871"    # 橙色
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})


# ==============================
# 读取 loss 数据
# ==============================
def loss_curve(name):
    """读取每个模型的loss记录"""
    df = pd.read_csv(name + "_loss_history.csv").values
    train_loss, test_loss = df[:, 0], df[:, 1]
    print(train_loss[-1], test_loss[-1])
    return np.arange(len(train_loss)), train_loss, test_loss


# ==============================
# 读取模型配置信息
# ==============================
def show_info(name):
    with open(name + "_config.json", 'r') as fp:
        return json.load(fp)


# ==============================
# 模型文件名
# ==============================
MODELS = [
    "FullOrder_20251116_194447_MLP_deeponet_NoPhys_NoBatch_HD64_HL5_LR0.01_Sample0.3",
    "FullOrder_20251114_165617_MLP_deeponet_Phys_Batch_HD64_HL5_LR0.01_Pretrain50_Sample0.3",
]

TITLES = [
    "Model 5: Pure Data + Scheduler",
    "Model 6: Physics-Informed + Pretrain + Scheduler"
]


# ==============================
# 主程序入口
# ==============================
if __name__ == '__main__':
    results = []
    for model in MODELS:
        info = show_info(model)
        results.append(info)
        print(f"Model: {model}\nConfig: {info}\n")

    with plt.style.context(['science', 'no-latex']):
        fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=200, constrained_layout=True)
        axes = axes.flatten()

        for i, (ax, model) in enumerate(zip(axes, MODELS)):
            steps, train_loss, test_loss = loss_curve(model)

            # 绘制曲线
            ax.plot(steps, train_loss, '-', lw=1.2, color=COLORS["train"], label='Train Loss')
            ax.plot(steps, test_loss, '--', lw=1.2, color=COLORS["test"], label='Test Loss')

            # 对数y轴
            ax.set_yscale('log')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss Value')
            ax.set_title(TITLES[i])

            # 样式优化
            # ax.grid(True, which='both', ls='--', lw=0.4)

            # 自适应图例位置（防止重叠）
            ax.legend(frameon=False, handlelength=1.5)

        plt.show()
