import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

# ==========================
# 绘图模式: "single"（合并大图） 或 "separate"（独立8张图）
# ==========================
PLOT_MODE = "separate"   # 可选 "single" / "separate"

# ==========================
# 全局绘图参数（可调）
# ==========================
rcParams['font.family'] = 'Times New Roman'
LABEL_SIZE = 18
TICK_SIZE = 16
TITLE_SIZE = 20
BAR_ALPHA = 0.25
BAR_LINEWIDTH = 0.1
BAR_WIDTH = 0.25
ANNOTATE_SIZE = 12
VIEW_ELEV = 60
VIEW_AZIM = 58

COLORS = {
    "train": "#71B8FF",
    "test":  "#FFB871"
}

# ==========================
# 读取数据
# ==========================
df = pd.read_csv("POD-PINN Report.csv")

fields_error = ["P_error", "Ur_error", "Ut_error", "Uz_error"]
fields_corr = ["P_corr", "Ur_corr", "Ut_corr", "Uz_corr"]
Title = {"P_error": "AWRE-Pressure", "Ur_error": "AWRE-Radial Velocity", "Ut_error": "AWRE-Tangential Velocity", "Uz_error": "AWRE-Axial Velocity",
         "P_corr": "PCC-Pressure", "Ur_corr": "PCC-Radial Velocity", "Ut_corr": "PCC-Tangential Velocity", "Uz_corr": "PCC-Axial Velocity", }

omegas = sorted(df["omega"].unique())     # 5 values
qvs = sorted(df["qv"].unique())        # 5 values

omega_to_idx = {v: i for i, v in enumerate(omegas)}
qv_to_idx = {v: i for i, v in enumerate(qvs)}


# ==========================
# 绘制 5×5 3D 柱状图
# ax: 若未提供则创建新图
# ==========================
def plot_field_matrix(field_name, is_error=True, fig_title="", ax=None):
    create_new_fig = False
    if ax is None:
        fig = plt.figure(figsize=(7.5, 7.5))
        ax = fig.add_subplot(111, projection="3d")
        create_new_fig = True
    else:
        ax.set_title(fig_title, fontsize=TITLE_SIZE)

    # 数据准备
    X, Y, Z, C = [], [], [], []

    for _, row in df.iterrows():
        xi = omega_to_idx[row["omega"]]
        yi = qv_to_idx[row["qv"]]
        val = row[field_name] * 100

        X.append(xi)
        Y.append(yi)
        Z.append(val)
        C.append(COLORS["train"] if row["dataset"] == "Training Set" else COLORS["test"])

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    ax.bar3d(
        X, Y, np.zeros_like(Z),
        BAR_WIDTH, BAR_WIDTH, Z,
        color=C,
        alpha=BAR_ALPHA,
        linewidth=BAR_LINEWIDTH,
        shade=True
    )

    # 柱顶标注
    for xi, yi, zi in zip(X, Y, Z):
        if is_error:
            val_pct = max(1, int(zi))
            text = f"{val_pct}"
        else:
            text = f"{int(zi)}"

        ax.text(
            xi + BAR_WIDTH/2,
            yi + BAR_WIDTH/2,
            zi,
            text,
            fontsize=ANNOTATE_SIZE,
            ha="center", va="bottom", color="black"
        )

    # 坐标轴设置
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))

    ax.set_xticklabels(omegas, fontsize=TICK_SIZE)
    ax.set_yticklabels(qvs, fontsize=TICK_SIZE)

    # 设置z轴刻度标签字体大小
    ax.tick_params(axis='z', labelsize=TICK_SIZE)

    ax.set_xlabel("rotating speed / rpm", fontsize=LABEL_SIZE, labelpad=12)
    ax.set_ylabel("flow rate / m³·s⁻¹", fontsize=LABEL_SIZE, labelpad=12)

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_title(fig_title, fontsize=TITLE_SIZE)

    # 图例（训练/测试）
    legend_elements = [
        Patch(facecolor=COLORS["train"], label="Training Set", alpha=BAR_ALPHA),
        Patch(facecolor=COLORS["test"],  label="Test Set",     alpha=BAR_ALPHA)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    if create_new_fig:
        plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
        plt.show()
    #plt.savefig(fig_title, dpi=400)


# ==========================
# 生成图：8 张独立（separate） 或 2×4 大图（single）
# ==========================
if PLOT_MODE == "separate":

    for f in fields_error:
        plot_field_matrix(f, is_error=True, fig_title=Title[f])

    for f in fields_corr:
        plot_field_matrix(f, is_error=False, fig_title=Title[f])

else:
    # ==========================
    # 合成大图（2 × 4）
    # ==========================
    fig = plt.figure(figsize=(15, 8))

    all_fields = fields_error + fields_corr

    for i, f in enumerate(all_fields):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        is_err = (i < 4)
        plot_field_matrix(f, is_error=is_err,ax=ax)
                          #fig_title=(f"AWRE of {f}" if is_err else f"PCC of {f}")

    plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
    plt.show()