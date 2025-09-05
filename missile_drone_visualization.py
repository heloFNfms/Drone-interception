import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
import matplotlib.widgets as widgets

# 设置中文字体和高质量渲染
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 创建图形（更大尺寸，更好效果，启用交互功能）
# 设置全屏大小的画布
fig = plt.figure(figsize=(24, 18))
# 最大化窗口
manager = plt.get_current_fig_manager()
try:
    manager.window.state('zoomed')  # Windows
except:
    try:
        manager.full_screen_toggle()  # 其他系统
    except:
        pass  # 如果都不支持就保持默认
fig.patch.set_facecolor('black')  # 黑色背景
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# 启用交互式导航工具栏
plt.rcParams['toolbar'] = 'toolmanager'

# 设置默认鼠标行为为平移而非旋转
plt.rcParams['keymap.pan'] = ['p']  # p键启用平移模式
plt.rcParams['keymap.zoom'] = ['o']  # o键启用缩放模式
plt.rcParams['keymap.home'] = ['h', 'r', 'home']  # 重置视图
plt.rcParams['keymap.back'] = ['left', 'c', 'backspace']  # 后退
plt.rcParams['keymap.forward'] = ['right', 'v']  # 前进

# 启用平移和缩放功能
plt.rcParams['keymap.pan'] = ['p']  # p键启用平移模式
plt.rcParams['keymap.zoom'] = ['o']  # o键启用缩放模式
plt.rcParams['keymap.home'] = ['h', 'r', 'home']  # 重置视图
plt.rcParams['keymap.back'] = ['left', 'c', 'backspace']  # 后退
plt.rcParams['keymap.forward'] = ['right', 'v']  # 前进

# 定义坐标
missile_pos = np.array([20000, 0, 2000])
drone_pos = np.array([17800, 0, 1800])
origin = np.array([0, 0, 0])

# 计算无人机飞行和烟雾弹轨迹
drone_speed = 120  # m/s
flight_time = 1.5  # s
explosion_delay = 3.6  # s
g = 9.8  # 重力加速度 m/s²

# 无人机飞行方向（朝向原点）
direction_to_origin = origin - drone_pos
direction_to_origin[2] = 0  # 等高度飞行，z方向不变
direction_unit = direction_to_origin / np.linalg.norm(direction_to_origin)

# 无人机1.5秒后的位置（投弹位置）
drop_distance = drone_speed * flight_time
drop_position = drone_pos + direction_unit * drop_distance
drop_position[2] = drone_pos[2]  # 保持等高度

# 烟雾弹初始速度（完全继承无人机的3D速度向量）
# 无人机以120m/s在水平方向朝向原点飞行，z方向速度为0
smoke_initial_velocity = np.array([
    direction_unit[0] * drone_speed,  # x方向速度
    direction_unit[1] * drone_speed,  # y方向速度  
    0.0                               # z方向初始速度为0（等高度飞行）
])

print(f"🚁 无人机飞行方向单位向量: {direction_unit}")
print(f"💣 烟雾弹初始速度向量: {smoke_initial_velocity} m/s")

# 烟雾弹爆炸位置计算（考虑初速度和重力）
# x, y方向：匀速直线运动
explosion_x = drop_position[0] + smoke_initial_velocity[0] * explosion_delay
explosion_y = drop_position[1] + smoke_initial_velocity[1] * explosion_delay
# z方向：自由落体运动
explosion_z = drop_position[2] + smoke_initial_velocity[2] * explosion_delay - 0.5 * g * explosion_delay**2

explosion_position = np.array([explosion_x, explosion_y, explosion_z])

# 计算导弹在烟雾弹爆炸时的位置
# 导弹总飞行时间 = 无人机飞行时间 + 烟雾弹飞行时间
total_time = flight_time + explosion_delay  # 1.5 + 3.6 = 5.1秒
missile_speed = 300  # m/s
missile_direction = (origin - missile_pos) / np.linalg.norm(origin - missile_pos)
missile_distance_traveled = missile_speed * total_time
missile_explosion_time_pos = missile_pos + missile_direction * missile_distance_traveled

# 高级颜色定义（渐变和发光效果）
colors = {
    'missile': '#FF2D2D',
    'missile_glow': '#FF6B6B',
    'drone': '#2D7FFF',
    'drone_glow': '#6BA3FF',
    'trajectory': '#FF4D4D',
    'trajectory_glow': '#FF8080',
    'origin': '#00FF88',
    'origin_glow': '#66FFAA',
    'grid': '#444444',
    'shadow': '#666666',
    'background': '#0A0A0A',
    'text': '#FFFFFF',
    'accent': '#FFD700',
    'smoke': '#AA44FF',
    'smoke_glow': '#CC77FF',
    'drop': '#FF8844',
    'drop_glow': '#FFAA66',
    'missile_time': '#FF9999',
    'missile_time_glow': '#FFBBBB',
    'target': '#00DD44',
    'target_glow': '#44FF77',
    'smoke_ball': '#DD44DD',
    'smoke_ball_glow': '#FF77FF'
}

# 绘制导弹轨迹（发光效果）
trajectory_points = np.linspace(missile_pos, origin, 200)
# 主轨迹线
ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
        color=colors['trajectory'], linestyle='-', linewidth=4, alpha=0.9, label='导弹轨迹')
# 发光效果
ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
        color=colors['trajectory_glow'], linestyle='-', linewidth=8, alpha=0.3)
ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
        color=colors['trajectory_glow'], linestyle='-', linewidth=12, alpha=0.1)

# 绘制导弹（优化大小）
def draw_missile_3d(ax, pos, color, size=100):
    """绘制3D导弹形状"""
    x, y, z = pos
    
    # 导弹主体（圆锥+圆柱）
    # 圆锥头部
    cone_height = size * 0.8
    cone_radius = size * 0.3
    
    # 圆柱体身
    cylinder_height = size * 1.5
    cylinder_radius = size * 0.2
    
    # 绘制圆锥头部
    theta = np.linspace(0, 2*np.pi, 20)
    
    # 圆锥侧面
    for i in range(len(theta)-1):
        cone_x = [x, x + cone_radius * np.cos(theta[i]), x + cone_radius * np.cos(theta[i+1])]
        cone_y = [y, y + cone_radius * np.sin(theta[i]), y + cone_radius * np.sin(theta[i+1])]
        cone_z = [z + cone_height, z, z]
        ax.plot_trisurf(cone_x, cone_y, cone_z, color=color, alpha=0.8)
    
    # 圆柱体身
    cylinder_x = x + cylinder_radius * np.cos(theta)
    cylinder_y = y + cylinder_radius * np.sin(theta)
    
    # 圆柱体顶面
    cylinder_z_top = np.full_like(cylinder_x, z)
    ax.plot(cylinder_x, cylinder_y, cylinder_z_top, color=color, linewidth=2)
    
    # 圆柱体底面
    cylinder_z_bottom = np.full_like(cylinder_x, z - cylinder_height)
    ax.plot(cylinder_x, cylinder_y, cylinder_z_bottom, color=color, linewidth=2)
    
    # 圆柱体侧面线条
    for i in range(0, len(theta), 4):
        ax.plot([cylinder_x[i], cylinder_x[i]], 
                [cylinder_y[i], cylinder_y[i]], 
                [z, z - cylinder_height], color=color, linewidth=1.5)
    
    # 主标记点（优化大小）
    ax.scatter([x], [y], [z], s=200, c=color, marker='^', 
              edgecolors=colors['missile_glow'], linewidth=2, alpha=1.0)
    # 发光光晕（减小）
    ax.scatter([x], [y], [z], s=400, c=colors['missile_glow'], marker='^', 
              alpha=0.2)
    ax.scatter([x], [y], [z], s=600, c=colors['missile_glow'], marker='^', 
              alpha=0.1)

# 绘制无人机（优化大小）
def draw_drone_3d(ax, pos, color, size=80):
    """绘制3D无人机形状"""
    x, y, z = pos
    
    # 机身主体
    body_length = size
    body_width = size * 0.4
    body_height = size * 0.2
    
    # 机身框架
    vertices = [
        [x - body_length/2, y - body_width/2, z - body_height/2],
        [x + body_length/2, y - body_width/2, z - body_height/2],
        [x + body_length/2, y + body_width/2, z - body_height/2],
        [x - body_length/2, y + body_width/2, z - body_height/2],
        [x - body_length/2, y - body_width/2, z + body_height/2],
        [x + body_length/2, y - body_width/2, z + body_height/2],
        [x + body_length/2, y + body_width/2, z + body_height/2],
        [x - body_length/2, y + body_width/2, z + body_height/2]
    ]
    
    # 绘制机边框
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
    ]
    
    for edge in edges:
        points = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot3D(*zip(*points), color=color, linewidth=2.5, alpha=0.8)
    
    # 螺旋桨
    prop_size = size * 0.6
    prop_positions = [
        [x - body_length/3, y - body_width*1.2, z + body_height],
        [x + body_length/3, y - body_width*1.2, z + body_height],
        [x - body_length/3, y + body_width*1.2, z + body_height],
        [x + body_length/3, y + body_width*1.2, z + body_height]
    ]
    
    for prop_pos in prop_positions:
        # 螺旋桨叶片
        ax.plot([prop_pos[0] - prop_size/2, prop_pos[0] + prop_size/2],
                [prop_pos[1], prop_pos[1]],
                [prop_pos[2], prop_pos[2]], color=color, linewidth=3)
        ax.plot([prop_pos[0], prop_pos[0]],
                [prop_pos[1] - prop_size/2, prop_pos[1] + prop_size/2],
                [prop_pos[2], prop_pos[2]], color=color, linewidth=3)
        
        # 螺旋桨中心
        ax.scatter(*prop_pos, s=30, c=color, marker='o')
    
    # 主标记点（优化大小）
    ax.scatter([x], [y], [z], s=200, c=color, marker='s', 
              edgecolors=colors['drone_glow'], linewidth=2, alpha=1.0)
    # 发光光晕（减小）
    ax.scatter([x], [y], [z], s=400, c=colors['drone_glow'], marker='s', 
              alpha=0.2)
    ax.scatter([x], [y], [z], s=600, c=colors['drone_glow'], marker='s', 
              alpha=0.1)

# 绘制导弹和无人机
draw_missile_3d(ax, missile_pos, colors['missile'])
draw_drone_3d(ax, drone_pos, colors['drone'])

# 绘制无人机飞行轨迹
flight_points = np.linspace(drone_pos, drop_position, 50)
ax.plot(flight_points[:, 0], flight_points[:, 1], flight_points[:, 2], 
        color=colors['drone'], linestyle='-', linewidth=3, alpha=0.8, label='无人机飞行轨迹')
# 发光效果
ax.plot(flight_points[:, 0], flight_points[:, 1], flight_points[:, 2], 
        color=colors['drone_glow'], linestyle='-', linewidth=6, alpha=0.3)

# 绘制投弹位置（优化大小）
ax.scatter(*drop_position, s=250, c=colors['drop'], marker='o', 
          edgecolors=colors['drop_glow'], linewidth=2, label='投弹位置', alpha=1.0)
# 投弹位置发光光晕（减小）
ax.scatter(*drop_position, s=500, c=colors['drop_glow'], marker='o', alpha=0.3)
ax.scatter(*drop_position, s=750, c=colors['drop_glow'], marker='o', alpha=0.15)

# 绘制烟雾弹轨迹（抛物线）
time_points = np.linspace(0, explosion_delay, 100)
smoke_trajectory = []
for t in time_points:
    x = drop_position[0] + smoke_initial_velocity[0] * t
    y = drop_position[1] + smoke_initial_velocity[1] * t
    z = drop_position[2] + smoke_initial_velocity[2] * t - 0.5 * g * t**2
    smoke_trajectory.append([x, y, z])

smoke_trajectory = np.array(smoke_trajectory)
ax.plot(smoke_trajectory[:, 0], smoke_trajectory[:, 1], smoke_trajectory[:, 2], 
        color=colors['smoke'], linestyle='--', linewidth=3, alpha=0.8, label='烟雾弹轨迹')
# 发光效果
ax.plot(smoke_trajectory[:, 0], smoke_trajectory[:, 1], smoke_trajectory[:, 2], 
        color=colors['smoke_glow'], linestyle='--', linewidth=6, alpha=0.3)

# 绘制烟雾弹爆炸位置（优化大小）
ax.scatter(*explosion_position, s=300, c=colors['smoke'], marker='*', 
          edgecolors=colors['smoke_glow'], linewidth=2, label='烟雾弹爆炸位置', alpha=1.0)
# 爆炸位置发光光晕（减小）
ax.scatter(*explosion_position, s=600, c=colors['smoke_glow'], marker='*', alpha=0.3)
ax.scatter(*explosion_position, s=900, c=colors['smoke_glow'], marker='*', alpha=0.15)
ax.scatter(*explosion_position, s=1200, c=colors['smoke_glow'], marker='*', alpha=0.08)

# 绘制导弹在爆炸时刻的位置
ax.scatter(*missile_explosion_time_pos, s=250, c=colors['missile_time'], marker='^', 
          edgecolors=colors['missile_time_glow'], linewidth=2, label='导弹爆炸时位置', alpha=1.0)
# 导弹爆炸时位置发光光晕
ax.scatter(*missile_explosion_time_pos, s=500, c=colors['missile_time_glow'], marker='^', alpha=0.3)
ax.scatter(*missile_explosion_time_pos, s=750, c=colors['missile_time_glow'], marker='^', alpha=0.15)

# 绘制导弹在爆炸时刻的轨迹段
missile_explosion_trajectory = np.linspace(missile_pos, missile_explosion_time_pos, 100)
ax.plot(missile_explosion_trajectory[:, 0], missile_explosion_trajectory[:, 1], missile_explosion_trajectory[:, 2], 
        color=colors['missile_time'], linestyle='-', linewidth=3, alpha=0.7, label='导弹5.1s轨迹')
# 发光效果
ax.plot(missile_explosion_trajectory[:, 0], missile_explosion_trajectory[:, 1], missile_explosion_trajectory[:, 2], 
        color=colors['missile_time_glow'], linestyle='-', linewidth=6, alpha=0.2)

# 绘制真目标圆柱体
def draw_target_cylinder(ax, center, radius, height, color):
    """绘制真目标圆柱体"""
    x0, y0, z0 = center
    
    # 创建圆柱体的侧面
    theta = np.linspace(0, 2*np.pi, 30)
    z_cylinder = np.linspace(z0, z0 + height, 10)
    
    # 侧面网格
    THETA, Z = np.meshgrid(theta, z_cylinder)
    X = x0 + radius * np.cos(THETA)
    Y = y0 + radius * np.sin(THETA)
    
    # 绘制侧面
    ax.plot_surface(X, Y, Z, color=color, alpha=0.7, linewidth=0.5, edgecolor='black')
    
    # 底面和顶面
    theta_circle = np.linspace(0, 2*np.pi, 30)
    r_circle = np.linspace(0, radius, 10)
    THETA_CIRCLE, R_CIRCLE = np.meshgrid(theta_circle, r_circle)
    
    # 底面
    X_bottom = x0 + R_CIRCLE * np.cos(THETA_CIRCLE)
    Y_bottom = y0 + R_CIRCLE * np.sin(THETA_CIRCLE)
    Z_bottom = np.full_like(X_bottom, z0)
    ax.plot_surface(X_bottom, Y_bottom, Z_bottom, color=color, alpha=0.7)
    
    # 顶面
    X_top = x0 + R_CIRCLE * np.cos(THETA_CIRCLE)
    Y_top = y0 + R_CIRCLE * np.sin(THETA_CIRCLE)
    Z_top = np.full_like(X_top, z0 + height)
    ax.plot_surface(X_top, Y_top, Z_top, color=color, alpha=0.7)
    
    # 添加标记点
    ax.scatter([x0], [y0], [z0 + height/2], s=200, c=color, marker='s', 
              edgecolors=colors['target_glow'], linewidth=2, alpha=1.0)

# 真目标位置和参数
target_center = np.array([0, 200, 0])
target_radius = 7  # 米
target_height = 10  # 米

# 绘制真目标圆柱体
draw_target_cylinder(ax, target_center, target_radius, target_height, colors['target'])

# 烟雾球参数
smoke_ball_radius = 10  # 米
smoke_sink_speed = 3  # m/s

# 计算烟雾球遮挡时间
def calculate_blocking_time():
    """计算烟雾球为圆柱体遮挡导弹的时间"""
    # 从爆炸时刻开始计算
    start_time = total_time  # 5.1秒
    blocking_start = None
    blocking_end = None
    
    # 修正：导弹飞向假目标(原点)，烟雾保护真目标
    missile_to_fake_target_direction = (origin - missile_pos) / np.linalg.norm(origin - missile_pos)
    
    print(f"\n🔍 遮挡计算详细分析:")
    print(f"导弹初始位置: {missile_pos}")
    print(f"假目标位置(导弹目标): {origin}")
    print(f"真目标位置(需保护): {target_center}")
    print(f"烟雾弹爆炸位置: {explosion_position}")
    print(f"导弹到假目标距离: {np.linalg.norm(origin - missile_pos):.1f}m")
    print(f"导弹到真目标距离: {np.linalg.norm(target_center - missile_pos):.1f}m")
    
    # 检查从爆炸后0到20秒的时间范围
    for t in np.arange(0, 20, 0.1):  # 每0.1秒检查一次
        current_time = start_time + t
        
        # 烟雾球当前位置（下沉）
        smoke_ball_pos = explosion_position.copy()
        smoke_ball_pos[2] -= smoke_sink_speed * t
        
        # 烟雾球如果下沉到地面以下，停止计算
        if smoke_ball_pos[2] < -smoke_ball_radius:
            break
        
        # 检查烟雾是否已超过有效时间（20秒）
        if t > 20:
            print(f"烟雾在{current_time:.1f}s时失效（超过20秒有效期）")
            if blocking_start is not None and blocking_end is None:
                blocking_end = current_time
                print(f"遮挡因烟雾失效而结束: {current_time:.1f}s")
            break
        
        # 导弹当前位置（修正：导弹飞向假目标原点）
        missile_current_pos = missile_pos + missile_to_fake_target_direction * missile_speed * current_time
        
        # 检查导弹是否已到达假目标
        if np.linalg.norm(missile_current_pos - origin) < 50:  # 导弹到达假目标附近
            print(f"导弹在{current_time:.1f}s时到达假目标")
            break
            
        # 检查烟雾球是否遮挡了导弹观察真目标圆柱体的视线（只有在有效时间内）
        if t <= 20 and is_blocking_cylinder_target(missile_current_pos, target_center, target_radius, target_height, smoke_ball_pos, smoke_ball_radius):
            if blocking_start is None:
                blocking_start = current_time
                print(f"遮挡开始: {current_time:.1f}s, 导弹位置: {missile_current_pos}, 烟雾球位置: {smoke_ball_pos}")
        else:
            if blocking_start is not None and blocking_end is None:
                blocking_end = current_time
                print(f"遮挡结束: {current_time:.1f}s, 导弹位置: {missile_current_pos}, 烟雾球位置: {smoke_ball_pos}")
                break
    
    # 如果遮挡一直持续到检查结束或烟雾失效
    if blocking_start is not None and blocking_end is None:
        blocking_end = start_time + 20  # 烟雾最大有效时间
        print(f"遮挡持续到烟雾失效: {blocking_end:.1f}s")
    
    return blocking_start, blocking_end

def is_blocking_cylinder_target(missile_pos, target_base_center, target_radius, target_height, smoke_pos, smoke_radius):
    """判断烟雾球是否遮挡导弹观察圆柱体目标的视线（改进版）"""
    # 圆柱体轴线：从底面中心到顶面中心
    target_bottom = target_base_center
    target_top = target_base_center + np.array([0, 0, target_height])
    
    # 导弹到目标的距离检查
    missile_to_target_dist = np.linalg.norm(target_base_center - missile_pos)
    if missile_to_target_dist < 100:  # 导弹距离目标太近
        return False
    
    # 在圆柱轴线上找与导弹z高度最接近的点
    missile_z = missile_pos[2]
    target_z_clamped = np.clip(missile_z, target_bottom[2], target_top[2])
    
    # 圆柱轴线上的最近点
    closest_point_on_axis = np.array([target_base_center[0], target_base_center[1], target_z_clamped])
    
    # 导弹到轴线最近点的视线方向
    missile_to_target = closest_point_on_axis - missile_pos
    missile_to_target_dist = np.linalg.norm(missile_to_target)
    
    if missile_to_target_dist < 1e-6:  # 避免除零
        return False
        
    missile_to_target_unit = missile_to_target / missile_to_target_dist
    
    # 导弹到烟雾球中心的向量
    missile_to_smoke = smoke_pos - missile_pos
    
    # 烟雾球中心在导弹-目标视线上的投影距离
    proj_dist = np.dot(missile_to_smoke, missile_to_target_unit)
    
    # 如果投影在导弹后方或目标后方，则不遮挡
    if proj_dist < 0 or proj_dist > missile_to_target_dist:
        return False
    
    # 计算烟雾球中心到导弹-目标视线的距离
    proj_point = missile_pos + proj_dist * missile_to_target_unit
    dist_to_line = np.linalg.norm(smoke_pos - proj_point)
    
    # 烟雾球的有效遮挡半径（考虑目标半径）
    effective_blocking_radius = smoke_radius + target_radius
    
    # 判断烟雾球是否能遮挡导弹观察目标的视线
    is_blocked = dist_to_line < effective_blocking_radius
    
    if is_blocked:
        print(f"    🎯 圆柱体遮挡检测: 导弹{missile_pos} -> 目标轴线点{closest_point_on_axis}")
        print(f"    💨 烟雾球位置: {smoke_pos}, 到视线距离: {dist_to_line:.1f}m")
        print(f"    🛡️ 有效遮挡半径: {effective_blocking_radius:.1f}m, 遮挡: {is_blocked}")
    
    return is_blocked

def is_blocking_real_target(missile_pos, real_target_pos, smoke_pos, smoke_radius, target_radius):
    """判断烟雾球是否遮挡导弹观察真目标的视线"""
    # 导弹到真目标的方向向量
    missile_to_real_target = real_target_pos - missile_pos
    missile_to_real_target_dist = np.linalg.norm(missile_to_real_target)
    
    if missile_to_real_target_dist < 100:  # 导弹距离真目标太近
        return False
        
    missile_to_real_target_unit = missile_to_real_target / missile_to_real_target_dist
    
    # 导弹到烟雾球中心的向量
    missile_to_smoke = smoke_pos - missile_pos
    
    # 烟雾球中心在导弹-真目标连线上的投影距离
    proj_dist = np.dot(missile_to_smoke, missile_to_real_target_unit)
    
    # 如果投影在导弹后方或真目标后方，则不遮挡
    if proj_dist < 0 or proj_dist > missile_to_real_target_dist:
        return False
    
    # 计算烟雾球中心到导弹-真目标连线的距离
    proj_point = missile_pos + proj_dist * missile_to_real_target_unit
    dist_to_line = np.linalg.norm(smoke_pos - proj_point)
    
    # 烟雾球的有效遮挡半径（考虑目标尺寸）
    effective_blocking_radius = smoke_radius + target_radius
    
    # 判断烟雾球是否能遮挡导弹观察真目标的视线
    is_blocked = dist_to_line < effective_blocking_radius
    
    if is_blocked:
        print(f"    遮挡检测: 导弹{missile_pos} -> 真目标{real_target_pos}")
        print(f"    烟雾球位置: {smoke_pos}, 到视线距离: {dist_to_line:.1f}m")
        print(f"    有效遮挡半径: {effective_blocking_radius:.1f}m, 遮挡: {is_blocked}")
    
    return is_blocked

# 计算遮挡时间
# 绘制遮挡关键时刻的位置
blocking_start, blocking_end = calculate_blocking_time()

if blocking_start is not None and blocking_end is not None:
    # 计算遮挡开始时刻的位置
    start_t = blocking_start - total_time  # 相对于爆炸时刻的时间
    start_missile_pos = missile_pos + missile_direction * missile_speed * blocking_start
    start_smoke_pos = explosion_position.copy()
    start_smoke_pos[2] -= smoke_sink_speed * start_t
    
    # 计算遮挡结束时刻的位置
    end_t = blocking_end - total_time  # 相对于爆炸时刻的时间
    end_missile_pos = missile_pos + missile_direction * missile_speed * blocking_end
    end_smoke_pos = explosion_position.copy()
    end_smoke_pos[2] -= smoke_sink_speed * end_t
    
    # 绘制遮挡开始时刻
    ax.scatter(*start_missile_pos, s=400, c='#FF0000', marker='D', 
              edgecolors='white', linewidth=3, label=f'导弹遮挡开始({blocking_start:.1f}s)', alpha=0.9)
    ax.scatter(*start_smoke_pos, s=400, c='#8B00FF', marker='o', 
              edgecolors='white', linewidth=3, label=f'烟雾球遮挡开始({blocking_start:.1f}s)', alpha=0.9)
    
    # 绘制遮挡结束时刻
    ax.scatter(*end_missile_pos, s=400, c='#FF6600', marker='D', 
              edgecolors='white', linewidth=3, label=f'导弹遮挡结束({blocking_end:.1f}s)', alpha=0.9)
    ax.scatter(*end_smoke_pos, s=400, c='#FF1493', marker='o', 
              edgecolors='white', linewidth=3, label=f'烟雾球遮挡结束({blocking_end:.1f}s)', alpha=0.9)
    
    # 绘制遮挡开始和结束的连线
    ax.plot([start_missile_pos[0], start_smoke_pos[0]], 
            [start_missile_pos[1], start_smoke_pos[1]], 
            [start_missile_pos[2], start_smoke_pos[2]], 
            color='#FF0000', linewidth=3, alpha=0.7, linestyle='--', label='遮挡开始视线')
    
    ax.plot([end_missile_pos[0], end_smoke_pos[0]], 
            [end_missile_pos[1], end_smoke_pos[1]], 
            [end_missile_pos[2], end_smoke_pos[2]], 
            color='#FF6600', linewidth=3, alpha=0.7, linestyle='--', label='遮挡结束视线')
    
    # 添加文本标注
    ax.text(start_missile_pos[0], start_missile_pos[1], start_missile_pos[2] + 50, 
            f'遮挡开始\n{blocking_start:.1f}s', fontsize=10, fontweight='bold',
            ha='center', va='bottom', color='#FF0000',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#FF0000', alpha=0.8, linewidth=2))
    
    ax.text(end_missile_pos[0], end_missile_pos[1], end_missile_pos[2] + 50, 
            f'遮挡结束\n{blocking_end:.1f}s', fontsize=10, fontweight='bold',
            ha='center', va='bottom', color='#FF6600',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#FF6600', alpha=0.8, linewidth=2))

# 绘制烟雾球在几个关键时刻的位置
key_times = [0, 5, 10, 15]  # 爆炸后的时间点
for i, t in enumerate(key_times):
    if t <= 20:  # 烟雾有效时间内
        smoke_ball_pos = explosion_position.copy()
        smoke_ball_pos[2] -= smoke_sink_speed * t
        
        # 只绘制在视角范围内的烟雾球
        if smoke_ball_pos[2] >= 1600:
            alpha = 0.6 - i * 0.1  # 随时间变淡
            
            # 绘制烟雾球（球体）
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = smoke_ball_pos[0] + smoke_ball_radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = smoke_ball_pos[1] + smoke_ball_radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = smoke_ball_pos[2] + smoke_ball_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                          color=colors['smoke_ball'], alpha=alpha, linewidth=0)
            
            if i == 0:  # 只为第一个添加标签
                ax.scatter(*smoke_ball_pos, s=100, c=colors['smoke_ball'], 
                          marker='o', label=f'烟雾球(t+{t}s)', alpha=alpha)

# 原点在当前视角范围外，添加指示箭头
# 添加指向原点的箭头
arrow_start_pos = np.array([16500, 0, 1800])
arrow_direction = (origin - arrow_start_pos) / np.linalg.norm(origin - arrow_start_pos) * 800
ax.quiver(arrow_start_pos[0], arrow_start_pos[1], arrow_start_pos[2],
          arrow_direction[0], arrow_direction[1], arrow_direction[2],
          color=colors['origin'], arrow_length_ratio=0.15, linewidth=4, alpha=0.8)
ax.text(arrow_start_pos[0], arrow_start_pos[1], arrow_start_pos[2] + 100, 
        '→ 目标原点', fontsize=11, fontweight='bold',
        ha='center', va='center', color=colors['origin'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['background'], 
                 edgecolor=colors['origin'], alpha=0.8, linewidth=1))

# 添加方向箭头
# 导弹到原点的方向
direction = origin - missile_pos
direction_norm = direction / np.linalg.norm(direction)
arrow_start = missile_pos + direction_norm * 500
arrow_length = direction_norm * 1000

ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
          arrow_length[0], arrow_length[1], arrow_length[2],
          color=colors['trajectory'], arrow_length_ratio=0.1, linewidth=3)

# 添加投影到地面（增强效果）
# 导弹投影
ax.plot([missile_pos[0], missile_pos[0]], [missile_pos[1], missile_pos[1]], [0, missile_pos[2]], 
        color=colors['missile'], linestyle=':', alpha=0.6, linewidth=3)
ax.scatter([missile_pos[0]], [missile_pos[1]], [0], s=150, c=colors['missile'], 
          marker='^', alpha=0.6, edgecolors=colors['missile_glow'])

# 无人机投影
ax.plot([drone_pos[0], drone_pos[0]], [drone_pos[1], drone_pos[1]], [0, drone_pos[2]], 
        color=colors['drone'], linestyle=':', alpha=0.6, linewidth=3)
ax.scatter([drone_pos[0]], [drone_pos[1]], [0], s=150, c=colors['drone'], 
          marker='s', alpha=0.6, edgecolors=colors['drone_glow'])

# 轨迹投影到地面（发光效果）
trajectory_ground = trajectory_points.copy()
trajectory_ground[:, 2] = 0
ax.plot(trajectory_ground[:, 0], trajectory_ground[:, 1], trajectory_ground[:, 2], 
        color=colors['trajectory'], linestyle=':', alpha=0.5, linewidth=3)
ax.plot(trajectory_ground[:, 0], trajectory_ground[:, 1], trajectory_ground[:, 2], 
        color=colors['trajectory_glow'], linestyle=':', alpha=0.2, linewidth=6)

# 绘制简化的地面网格（减少视觉干扰）
# 只在关键区域绘制网格
high_area_x = np.linspace(16000, 21000, 11)
high_area_y = np.linspace(-500, 500, 11)
X_high, Y_high = np.meshgrid(high_area_x, high_area_y)
Z_high = np.full_like(X_high, 1600)  # 在视角底部绘制网格
# 主网格（更透明）
ax.plot_wireframe(X_high, Y_high, Z_high, color=colors['grid'], alpha=0.2, linewidth=0.5)
# 发光网格（更淡）
ax.plot_wireframe(X_high, Y_high, Z_high, color=colors['missile_glow'], alpha=0.1, linewidth=0.8)

# 添加时间和距离标注（优化位置避免拥挤）
# 无人机飞行距离标注（放在轨迹上方）
flight_mid = (drone_pos + drop_position) / 2
flight_distance = np.linalg.norm(drop_position - drone_pos)
ax.text(flight_mid[0], flight_mid[1] + 300, flight_mid[2] + 100, 
        f'飞行{flight_time}s\n{flight_distance:.0f}m', fontsize=10, fontweight='bold',
        ha='center', va='center', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['drone'], 
                 edgecolor=colors['text'], alpha=0.7, linewidth=1))

# 烟雾弹飞行时间标注（放在轨迹下方）
smoke_mid = (drop_position + explosion_position) / 2
smoke_distance = np.linalg.norm(explosion_position - drop_position)
ax.text(smoke_mid[0], smoke_mid[1] - 300, smoke_mid[2] - 50, 
        f'爆炸{explosion_delay}s\n{smoke_distance:.0f}m', fontsize=10, fontweight='bold',
        ha='center', va='center', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['smoke'], 
                 edgecolor=colors['text'], alpha=0.7, linewidth=1))

# 导弹在爆炸时刻的位置标注
missile_time_mid = (missile_pos + missile_explosion_time_pos) / 2
missile_distance_at_explosion = np.linalg.norm(missile_explosion_time_pos - missile_pos)
ax.text(missile_time_mid[0], missile_time_mid[1] + 200, missile_time_mid[2] + 80, 
        f'导弹{total_time}s\n{missile_distance_at_explosion:.0f}m', fontsize=10, fontweight='bold',
        ha='center', va='center', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['missile_time'], 
                 edgecolor=colors['text'], alpha=0.7, linewidth=1))

# 添加距离标注（优化位置）
distance = np.linalg.norm(missile_pos - drone_pos)
mid_point = (missile_pos + drone_pos) / 2
ax.text(mid_point[0], mid_point[1], mid_point[2] + 150, 
        f'{distance:.0f}m', fontsize=11, fontweight='bold',
        ha='center', va='center', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['accent'], 
                 edgecolor=colors['text'], alpha=0.8, linewidth=1))

# 连接导弹和无人机的线（发光效果）
ax.plot([missile_pos[0], drone_pos[0]], 
        [missile_pos[1], drone_pos[1]], 
        [missile_pos[2], drone_pos[2]], 
        color=colors['accent'], linewidth=4, alpha=0.8, label='距离连线')
# 发光效果
ax.plot([missile_pos[0], drone_pos[0]], 
        [missile_pos[1], drone_pos[1]], 
        [missile_pos[2], drone_pos[2]], 
        color=colors['accent'], linewidth=8, alpha=0.3)
ax.plot([missile_pos[0], drone_pos[0]], 
        [missile_pos[1], drone_pos[1]], 
        [missile_pos[2], drone_pos[2]], 
        color=colors['accent'], linewidth=12, alpha=0.1)

# 设置坐标轴（高级样式）
ax.set_xlabel('X 坐标 (米)', fontsize=14, fontweight='bold', color=colors['text'])
ax.set_ylabel('Y 坐标 (米)', fontsize=14, fontweight='bold', color=colors['text'])
ax.set_zlabel('Z 坐标 (米)', fontsize=14, fontweight='bold', color=colors['text'])

# 设置坐标轴刻度颜色
ax.tick_params(axis='x', colors=colors['text'])
ax.tick_params(axis='y', colors=colors['text'])
ax.tick_params(axis='z', colors=colors['text'])

# 设置坐标轴范围（聚焦关键区域，减少拥挤）
ax.set_xlim(16000, 21000)  # 聚焦到导弹和无人机区域
ax.set_ylim(-500, 500)
ax.set_zlim(1600, 2200)    # 聚焦到飞行高度区域

# 设置标题（简化）
ax.set_title('导弹与无人机动态轨迹分析', 
            fontsize=16, fontweight='bold', pad=20, color=colors['text'])

# 添加图例（高级样式）
legend_elements = [
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['missile'], 
              markersize=12, label='导弹初始位置', markeredgecolor=colors['missile_glow']),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['missile_time'], 
              markersize=12, label='导弹爆炸时位置(5.1s)', markeredgecolor=colors['missile_time_glow']),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['drone'], 
              markersize=12, label='无人机初始位置', markeredgecolor=colors['drone_glow']),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['drop'], 
              markersize=12, label='投弹位置(1.5s后)', markeredgecolor=colors['drop_glow']),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['smoke'], 
              markersize=14, label='烟雾弹爆炸位置', markeredgecolor=colors['smoke_glow']),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['target'], 
              markersize=12, label='真目标圆柱体', markeredgecolor=colors['target_glow']),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['smoke_ball'], 
              markersize=10, label='烟雾球', markeredgecolor=colors['smoke_ball_glow']),
    plt.Line2D([0], [0], color=colors['trajectory'], linestyle='-', 
              linewidth=4, label='导弹完整轨迹'),
    plt.Line2D([0], [0], color=colors['missile_time'], linestyle='-', 
              linewidth=3, label='导弹5.1s轨迹'),
    plt.Line2D([0], [0], color=colors['drone'], linestyle='-', 
              linewidth=3, label='无人机飞行轨迹'),
    plt.Line2D([0], [0], color=colors['smoke'], linestyle='--', 
              linewidth=3, label='烟雾弹轨迹'),
    plt.Line2D([0], [0], color=colors['accent'], linewidth=4, label='距离连线')
]
# 优化图例布局（移到图形外部，避免遮挡）
legend = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 fontsize=9, framealpha=0.95, facecolor=colors['background'], 
                 edgecolor=colors['text'], labelcolor=colors['text'])
legend.get_frame().set_linewidth(1)
# 添加图例框的内边距
legend.get_frame().set_boxstyle('round,pad=0.3')

# 设置视角（增强立体感）
ax.view_init(elev=20, azim=45)

# 美化坐标轴（高级效果）
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor(colors['grid'])
ax.yaxis.pane.set_edgecolor(colors['grid'])
ax.zaxis.pane.set_edgecolor(colors['grid'])
ax.xaxis.pane.set_alpha(0.2)
ax.yaxis.pane.set_alpha(0.2)
ax.zaxis.pane.set_alpha(0.2)

# 设置坐标轴线条样式
ax.xaxis.line.set_color(colors['text'])
ax.yaxis.line.set_color(colors['text'])
ax.zaxis.line.set_color(colors['text'])

# 添加网格
ax.grid(True, alpha=0.3)

# 添加交互式缩放功能
def on_scroll(event):
    """鼠标滚轮缩放功能"""
    if event.inaxes != ax:
        return
    
    # 获取当前坐标轴范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    # 计算缩放因子
    scale_factor = 1.1 if event.button == 'up' else 0.9
    
    # 计算中心点
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    z_center = (zlim[0] + zlim[1]) / 2
    
    # 计算新的范围
    x_range = (xlim[1] - xlim[0]) * scale_factor / 2
    y_range = (ylim[1] - ylim[0]) * scale_factor / 2
    z_range = (zlim[1] - zlim[0]) * scale_factor / 2
    
    # 设置新的坐标轴范围
    ax.set_xlim(x_center - x_range, x_center + x_range)
    ax.set_ylim(y_center - y_range, y_center + y_range)
    ax.set_zlim(z_center - z_range, z_center + z_range)
    
    # 重新绘制
    fig.canvas.draw()

# 添加重置视图功能
def reset_view(event=None):
    """重置到默认视图"""
    ax.set_xlim(16000, 21000)
    ax.set_ylim(-500, 500)
    ax.set_zlim(1600, 2200)
    ax.view_init(elev=20, azim=45)
    fig.canvas.draw()

# 添加缩放按钮
ax_reset = plt.axes([0.02, 0.02, 0.08, 0.04])
ax_reset.patch.set_facecolor('black')
button_reset = Button(ax_reset, '重置视图', color='gray', hovercolor='lightgray')
button_reset.label.set_color('white')
button_reset.on_clicked(reset_view)

# 添加放大按钮
ax_zoom_in = plt.axes([0.11, 0.02, 0.06, 0.04])
ax_zoom_in.patch.set_facecolor('black')
button_zoom_in = Button(ax_zoom_in, '放大', color='gray', hovercolor='lightgray')
button_zoom_in.label.set_color('white')

def zoom_in(event):
    """放大功能"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    z_center = (zlim[0] + zlim[1]) / 2
    
    scale_factor = 0.8
    x_range = (xlim[1] - xlim[0]) * scale_factor / 2
    y_range = (ylim[1] - ylim[0]) * scale_factor / 2
    z_range = (zlim[1] - zlim[0]) * scale_factor / 2
    
    ax.set_xlim(x_center - x_range, x_center + x_range)
    ax.set_ylim(y_center - y_range, y_center + y_range)
    ax.set_zlim(z_center - z_range, z_center + z_range)
    fig.canvas.draw()

button_zoom_in.on_clicked(zoom_in)

# 添加缩小按钮
ax_zoom_out = plt.axes([0.18, 0.02, 0.06, 0.04])
ax_zoom_out.patch.set_facecolor('black')
button_zoom_out = Button(ax_zoom_out, '缩小', color='gray', hovercolor='lightgray')
button_zoom_out.label.set_color('white')

def zoom_out(event):
    """缩小功能"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    z_center = (zlim[0] + zlim[1]) / 2
    
    scale_factor = 1.25
    x_range = (xlim[1] - xlim[0]) * scale_factor / 2
    y_range = (ylim[1] - ylim[0]) * scale_factor / 2
    z_range = (zlim[1] - zlim[0]) * scale_factor / 2
    
    ax.set_xlim(x_center - x_range, x_center + x_range)
    ax.set_ylim(y_center - y_range, y_center + y_range)
    ax.set_zlim(z_center - z_range, z_center + z_range)
    fig.canvas.draw()

button_zoom_out.on_clicked(zoom_out)

# 连接鼠标滚轮事件
fig.canvas.mpl_connect('scroll_event', on_scroll)

# 添加键盘快捷键支持
def on_key_press(event):
    """键盘快捷键"""
    if event.key == '+':
        zoom_in(None)
    elif event.key == '-':
        zoom_out(None)
    elif event.key == 'r':
        reset_view(None)
    elif event.key == 'h':
        print("\n=== 交互操作说明 ===")
        print("鼠标滚轮: 放大/缩小")
        print("+ 键: 放大")
        print("- 键: 缩小")
        print("r 键: 重置视图")
        print("h 键: 显示帮助")
        print("鼠标拖拽: 旋转视角")
        print("==================")

fig.canvas.mpl_connect('key_press_event', on_key_press)

# 调整布局
plt.tight_layout()

# 显示操作提示
print("\n=== 交互操作说明 ===")
print("🖱️  鼠标滚轮: 放大/缩小")
print("⌨️  + 键: 放大")
print("⌨️  - 键: 缩小")
print("⌨️  r 键: 重置视图")
print("⌨️  h 键: 显示帮助")
print("🖱️  鼠标拖拽: 旋转视角")
print("🔘 按钮: 使用底部的放大/缩小/重置按钮")
print("==================\n")

# 显示图形
plt.show()

# 输出关键信息
print("=" * 70)
print("导弹与无人机动态轨迹分析")
print("=" * 70)
print(f"导弹初始位置: {tuple(missile_pos)}")
print(f"无人机初始位置: {tuple(drone_pos)}")
print(f"无人机飞行速度: {drone_speed} m/s")
print(f"无人机飞行时间: {flight_time} s")
print(f"投弹位置: ({drop_position[0]:.1f}, {drop_position[1]:.1f}, {drop_position[2]:.1f})")
print(f"烟雾弹爆炸延迟: {explosion_delay} s")
print(f"烟雾弹爆炸位置: ({explosion_position[0]:.1f}, {explosion_position[1]:.1f}, {explosion_position[2]:.1f})")
print(f"目标原点: {tuple(origin)}")
print("\n距离分析:")
print(f"导弹与无人机初始距离: {distance:.2f} 米")
print(f"导弹到原点距离: {np.linalg.norm(missile_pos - origin):.2f} 米")
print(f"无人机到原点距离: {np.linalg.norm(drone_pos - origin):.2f} 米")
print(f"投弹位置到原点距离: {np.linalg.norm(drop_position - origin):.2f} 米")
print(f"爆炸位置到原点距离: {np.linalg.norm(explosion_position - origin):.2f} 米")
print(f"导弹爆炸时位置: ({missile_explosion_time_pos[0]:.1f}, {missile_explosion_time_pos[1]:.1f}, {missile_explosion_time_pos[2]:.1f})")
print(f"导弹爆炸时到原点距离: {np.linalg.norm(missile_explosion_time_pos - origin):.2f} 米")
print(f"导弹与烟雾弹爆炸时距离: {np.linalg.norm(missile_explosion_time_pos - explosion_position):.2f} 米")
print(f"真目标圆柱体位置: {tuple(target_center)}")
print(f"真目标圆柱体参数: 半径{target_radius}m, 高度{target_height}m")
print(f"烟雾球半径: {smoke_ball_radius}m")
print(f"烟雾球下沉速度: {smoke_sink_speed}m/s")
print(f"烟雾有效时间: 20s（起爆后失效）")

if blocking_start is not None and blocking_end is not None:
    blocking_duration = blocking_end - blocking_start
    print(f"\n🛡️ 烟雾遮挡分析:")
    print(f"遮挡开始时间: {blocking_start:.1f}s")
    print(f"遮挡结束时间: {blocking_end:.1f}s")
    print(f"遮挡持续时间: {blocking_duration:.1f}s")
else:
    print(f"\n🛡️ 烟雾遮挡分析: 烟雾球未能有效遮挡导弹视线")

print("=" * 70)