# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体（确保系统中有支持的中文字体，如SimHei、微软雅黑等）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 定义所有坐标点
fake_target = np.array([0, 0, 0])  # 假目标（原点）

# 真目标参数
true_center = np.array([0, 200, 0])
true_radius = 7
true_height = 10

# 导弹初始位置
missiles = {
    "M1": np.array([20000, 0, 2000]),
    "M2": np.array([19000, 600, 2100]),
    "M3": np.array([18000, -600, 1900])
}

# 无人机初始位置
drones = {
    "FY1": np.array([17800, 0, 1800]),
    "FY2": np.array([12000, 1400, 1400]),
    "FY3": np.array([6000, -3000, 700]),
    "FY4": np.array([11000, 2000, 1800]),
    "FY5": np.array([13000, -2000, 1300])
}

# 2. 创建3D图形
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 3. 绘制假目标（红点+标签）
ax.scatter(*fake_target, c='red', s=150, marker='*', label='假目标（原点）')

# 4. 绘制真目标圆柱体
theta = np.linspace(0, 2*np.pi, 36)
# 底部圆（z=0）
x_bottom = true_center[0] + true_radius * np.cos(theta)
y_bottom = true_center[1] + true_radius * np.sin(theta)
z_bottom = np.zeros_like(theta)
ax.plot(x_bottom, y_bottom, z_bottom, 'g-', lw=4, label='真目标底部')

# 顶部圆（z=true_height）
x_top = true_center[0] + true_radius * np.cos(theta)
y_top = true_center[1] + true_radius * np.sin(theta)
z_top = true_height * np.ones_like(theta)
ax.plot(x_top, y_top, z_top, 'g-', lw=4)

# 连接母线（显示圆柱高度）
for z in np.linspace(0, true_height, 5):
    ax.plot(x_bottom, y_bottom, [z]*len(theta), 'g--', alpha=0.5)

# 5. 绘制导弹位置（蓝色点）
for name, pos in missiles.items():
    ax.scatter(*pos, c='blue', s=70, label=f'导弹 {name}')

# 6. 绘制无人机位置（青色点）
for name, pos in drones.items():
    ax.scatter(*pos, c='cyan', s=70, label=f'无人机 {name}')

# 7. 绘制烟幕云团示例（橙色半透明球）
def plot_smoke_cloud(ax, center, radius=10, alpha=0.4):
 """绘制烟幕云团的辅助函数"""
 u = np.linspace(0, 2*np.pi, 30)
 v = np.linspace(0, np.pi, 30)
 x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
 y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
 z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
 ax.plot_surface(x, y, z, color='orange', alpha=alpha)

# 示例：在FY1位置下方3m处画烟幕（模拟t=1s后）
smoke_center = drones["FY1"] - np.array([0, 0, 3])
plot_smoke_cloud(ax, smoke_center)

# 8. 设置图形属性
ax.set_xlabel('X 坐标 (米)', fontsize=12)
ax.set_ylabel('Y 坐标 (米)', fontsize=12)
ax.set_zlabel('Z 坐标 (米)', fontsize=12)
ax.set_title('烟幕干扰弹投放策略3D示意图', fontsize=14, pad=20)

# 调整视角（可根据需要修改）
ax.view_init(elev=25, azim=-70)

# 处理图例（避免重复标签）
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.show()