import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
import matplotlib.widgets as widgets

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 创建图形（更大尺寸，更好效果，启用交互功能）
fig = plt.figure(figsize=(24, 18))

# 窗口最大化
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

# 启用平移和缩放功能
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

# 初始位置设定
missile_pos = np.array([20000, 0, 2000])
drone_pos = np.array([17800, 0, 1800])
origin = np.array([0, 0, 0])

# 运动参数
drone_speed = 120  # m/s
flight_time = 1.5  # s
explosion_delay = 3.6  # s
g = 9.8  # 重力加速度 m/s²

# 计算无人机飞行方向（朝向原点，等高度飞行）
direction_to_origin = origin - drone_pos
direction_to_origin[2] = 0  # 等高度飞行，z方向不变
direction_unit = direction_to_origin / np.linalg.norm(direction_to_origin)

# 计算投弹位置（1.5秒后）
drop_distance = drone_speed * flight_time
drop_position = drone_pos + direction_unit * drop_distance
drop_position[2] = drone_pos[2]  # 保持等高度

# 烟雾弹初始速度（继承无人机速度）
smoke_initial_velocity = np.array([
    direction_unit[0] * drone_speed,  # x方向速度
    direction_unit[1] * drone_speed,  # y方向速度  
    0.0                               # z方向初始速度为0（等高度飞行）
])

print(f"🚁 无人机飞行方向单位向量: {direction_unit}")
print(f"💣 烟雾弹初始速度向量: {smoke_initial_velocity} m/s")

# 计算烟雾弹爆炸位置（投弹后3.6秒爆炸）
explosion_x = drop_position[0] + smoke_initial_velocity[0] * explosion_delay
explosion_y = drop_position[1] + smoke_initial_velocity[1] * explosion_delay
# z方向考虑重力影响
explosion_z = drop_position[2] + smoke_initial_velocity[2] * explosion_delay - 0.5 * g * explosion_delay**2

explosion_position = np.array([explosion_x, explosion_y, explosion_z])

# 导弹运动参数
total_time = flight_time + explosion_delay  # 1.5 + 3.6 = 5.1秒
missile_speed = 300  # m/s
missile_direction = (origin - missile_pos) / np.linalg.norm(origin - missile_pos)

# 真目标参数
target_center = np.array([0, 200, 0])
target_radius = 7  # 米
target_height = 10  # 米

# 烟雾球参数
smoke_ball_radius = 10  # 米
smoke_sink_speed = 3  # m/s

# 颜色配置
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

# 动画参数
animation_duration = 25.0  # 总动画时长（秒）
fps = 30  # 帧率
total_frames = int(animation_duration * fps)
time_step = animation_duration / total_frames

# 存储动画数据
animation_data = {
    'missile_positions': [],
    'drone_positions': [],
    'smoke_positions': [],
    'smoke_ball_positions': [],
    'times': [],
    'events': [],
    'blocking_status': []  # 遮挡状态
}

# 遮挡检测函数
def is_blocking_cylinder_target(missile_pos, target_base_center, target_radius, target_height, smoke_pos, smoke_radius):
    """判断烟雾球是否遮挡导弹观察圆柱体目标的视线（圆锥-圆柱体相交检测）"""
    # 导弹到烟雾球中心的向量
    missile_to_smoke = smoke_pos - missile_pos
    missile_to_smoke_dist = np.linalg.norm(missile_to_smoke)
    
    if missile_to_smoke_dist < 1e-6:  # 导弹与烟雾球重合
        return True
        
    # 导弹到烟雾球中心的单位向量
    missile_to_smoke_unit = missile_to_smoke / missile_to_smoke_dist
    
    # 计算从导弹到烟雾球的切线角度（半角）
    if missile_to_smoke_dist <= smoke_radius:  # 导弹在烟雾球内部
        return True
        
    # 切线半角的正弦值
    sin_half_angle = smoke_radius / missile_to_smoke_dist
    cos_half_angle = np.sqrt(1 - sin_half_angle**2)
    
    # 圆柱体参数
    cylinder_bottom = target_base_center
    cylinder_top = target_base_center + np.array([0, 0, target_height])
    cylinder_axis = np.array([0, 0, 1])  # 圆柱体轴线方向（垂直向上）
    
    # 检测圆锥是否与圆柱体相交
    # 方法：检测圆柱体轴线上的点是否在圆锥内
    
    # 导弹到圆柱体底面中心的向量
    missile_to_cylinder_bottom = cylinder_bottom - missile_pos
    missile_to_cylinder_top = cylinder_top - missile_pos
    
    # 检查圆柱体底面和顶面是否在圆锥内
    def point_in_cone(point_vec):
        point_dist = np.linalg.norm(point_vec)
        if point_dist < 1e-6:
            return True
        point_unit = point_vec / point_dist
        # 计算点与圆锥轴线的夹角余弦值
        cos_angle = np.dot(point_unit, missile_to_smoke_unit)
        return cos_angle >= cos_half_angle
    
    # 检查圆柱体底面和顶面中心是否在圆锥内
    bottom_in_cone = point_in_cone(missile_to_cylinder_bottom)
    top_in_cone = point_in_cone(missile_to_cylinder_top)
    
    if bottom_in_cone or top_in_cone:
        return True
    
    # 检查圆柱体边缘是否与圆锥相交
    # 在圆柱体轴线上采样多个点，检查这些点加上半径偏移后是否在圆锥内
    num_samples = 10
    for i in range(num_samples + 1):
        t = i / num_samples
        sample_center = cylinder_bottom + t * (cylinder_top - cylinder_bottom)
        
        # 在该高度的圆周上检查多个点
        num_circle_samples = 8
        for j in range(num_circle_samples):
            angle = 2 * np.pi * j / num_circle_samples
            # 圆柱体边缘点（在xy平面内的圆周上）
            edge_point = sample_center + target_radius * np.array([np.cos(angle), np.sin(angle), 0])
            edge_vec = edge_point - missile_pos
            
            if point_in_cone(edge_vec):
                return True
    
    return False

# 全局变量存储遮挡状态
blocking_start_time = None
blocking_end_time = None
blocking_detected = False
animation_should_stop = False

# 预计算所有帧的位置
for frame in range(total_frames):
    current_time = frame * time_step
    animation_data['times'].append(current_time)
    
    # 导弹位置（始终朝向原点匀速运动）
    missile_distance = missile_speed * current_time
    missile_current_pos = missile_pos + missile_direction * missile_distance
    animation_data['missile_positions'].append(missile_current_pos.copy())
    
    # 无人机位置
    if current_time <= flight_time:
        # 飞行阶段
        drone_distance = drone_speed * current_time
        drone_current_pos = drone_pos + direction_unit * drone_distance
        drone_current_pos[2] = drone_pos[2]  # 保持等高度
    else:
        # 投弹后保持在投弹位置
        drone_current_pos = drop_position.copy()
    animation_data['drone_positions'].append(drone_current_pos.copy())
    
    # 烟雾弹位置
    if current_time <= flight_time:
        # 投弹前，烟雾弹跟随无人机
        smoke_current_pos = drone_current_pos.copy()
    elif current_time <= total_time:
        # 投弹后到爆炸前，烟雾弹做抛物线运动
        t_smoke = current_time - flight_time
        smoke_x = drop_position[0] + smoke_initial_velocity[0] * t_smoke
        smoke_y = drop_position[1] + smoke_initial_velocity[1] * t_smoke
        smoke_z = drop_position[2] + smoke_initial_velocity[2] * t_smoke - 0.5 * g * t_smoke**2
        smoke_current_pos = np.array([smoke_x, smoke_y, smoke_z])
    else:
        # 爆炸后，烟雾弹消失
        smoke_current_pos = explosion_position.copy()
    animation_data['smoke_positions'].append(smoke_current_pos.copy())
    
    # 烟雾球位置（爆炸后出现并下沉）
    if current_time >= total_time:
        t_smoke_ball = current_time - total_time
        smoke_ball_z = explosion_position[2] - smoke_sink_speed * t_smoke_ball
        if smoke_ball_z >= 1600 and t_smoke_ball <= 20:  # 烟雾有效时间20秒
            smoke_ball_pos = np.array([explosion_position[0], explosion_position[1], smoke_ball_z])
        else:
            smoke_ball_pos = None
    else:
        smoke_ball_pos = None
    animation_data['smoke_ball_positions'].append(smoke_ball_pos)
    
    # 遮挡检测（仅在烟雾球存在时）
    is_blocked = False
    if smoke_ball_pos is not None:
        # 真目标参数
        target_center = np.array([0, 200, 0])
        target_radius = 7
        target_height = 10
        smoke_radius = 15  # 烟雾球半径
        
        is_blocked = is_blocking_cylinder_target(
            missile_pos, target_center, target_radius, target_height,
            smoke_ball_pos, smoke_radius
        )
    
    animation_data['blocking_status'].append(is_blocked)

# 绘制函数
def draw_missile_3d(ax, pos, color, size=100):
    """绘制3D导弹"""
    # 导弹主体（圆锥形）
    ax.scatter(*pos, s=size*2, c=color, marker='^', 
              edgecolors='white', linewidth=2, alpha=1.0)
    # 发光效果
    ax.scatter(*pos, s=size*4, c=color, marker='^', alpha=0.3)
    ax.scatter(*pos, s=size*6, c=color, marker='^', alpha=0.15)

def draw_drone_3d(ax, pos, color, size=80):
    """绘制3D无人机"""
    # 无人机主体
    ax.scatter(*pos, s=size*2, c=color, marker='s', 
              edgecolors='white', linewidth=2, alpha=1.0)
    # 发光效果
    ax.scatter(*pos, s=size*4, c=color, marker='s', alpha=0.3)
    ax.scatter(*pos, s=size*6, c=color, marker='s', alpha=0.15)

def draw_target_cylinder(ax, center, radius, height, color):
    """绘制目标圆柱体"""
    # 圆柱体参数
    theta = np.linspace(0, 2*np.pi, 20)
    z_cyl = np.linspace(center[2], center[2] + height, 10)
    
    # 侧面
    for z in z_cyl:
        x_circle = center[0] + radius * np.cos(theta)
        y_circle = center[1] + radius * np.sin(theta)
        z_circle = np.full_like(x_circle, z)
        ax.plot(x_circle, y_circle, z_circle, color=color, alpha=0.6, linewidth=2)
    
    # 顶面和底面
    for z in [center[2], center[2] + height]:
        x_circle = center[0] + radius * np.cos(theta)
        y_circle = center[1] + radius * np.sin(theta)
        z_circle = np.full_like(x_circle, z)
        ax.plot(x_circle, y_circle, z_circle, color=color, alpha=0.8, linewidth=3)

def draw_smoke_ball(ax, pos, radius, color, alpha=0.6):
    """绘制烟雾球"""
    if pos is None:
        return
    
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = pos[0] + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = pos[1] + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = pos[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                   color=color, alpha=alpha, linewidth=0)

# 初始化图形
def init_plot():
    ax.clear()
    ax.set_facecolor('black')
    
    # 设置坐标轴
    ax.set_xlim(16000, 21000)
    ax.set_ylim(-500, 500)
    ax.set_zlim(1600, 2200)
    
    ax.set_xlabel('X 坐标 (米)', fontsize=14, fontweight='bold', color=colors['text'])
    ax.set_ylabel('Y 坐标 (米)', fontsize=14, fontweight='bold', color=colors['text'])
    ax.set_zlabel('Z 坐标 (米)', fontsize=14, fontweight='bold', color=colors['text'])
    
    ax.tick_params(axis='x', colors=colors['text'])
    ax.tick_params(axis='y', colors=colors['text'])
    ax.tick_params(axis='z', colors=colors['text'])
    
    ax.set_title('导弹与无人机动态轨迹分析 - 实时仿真', 
                fontsize=16, fontweight='bold', pad=20, color=colors['text'])
    
    # 绘制静态元素
    # 目标圆柱体
    draw_target_cylinder(ax, target_center, target_radius, target_height, colors['target'])
    
    # 原点标记
    ax.scatter(*origin, s=300, c=colors['origin'], marker='*', 
              edgecolors=colors['origin_glow'], linewidth=2, alpha=1.0)
    
    # 导弹完整轨迹（虚线）
    trajectory_points = np.linspace(missile_pos, origin, 200)
    ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
            color=colors['trajectory'], linestyle=':', linewidth=2, alpha=0.5)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 设置面板样式
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(colors['grid'])
    ax.yaxis.pane.set_edgecolor(colors['grid'])
    ax.zaxis.pane.set_edgecolor(colors['grid'])
    ax.xaxis.pane.set_alpha(0.2)
    ax.yaxis.pane.set_alpha(0.2)
    ax.zaxis.pane.set_alpha(0.2)
    
    ax.grid(True, alpha=0.3)

# 动画更新函数
def animate(frame):
    global blocking_start_time, blocking_end_time, blocking_detected, animation_should_stop
    
    # 清除动态元素（保留静态元素）
    # 清除散点图和线条
    for collection in ax.collections[:]:
        if hasattr(collection, '_facecolors') or hasattr(collection, '_edgecolors'):
            collection.remove()
    
    current_time = animation_data['times'][frame]
    missile_pos_current = animation_data['missile_positions'][frame]
    drone_pos_current = animation_data['drone_positions'][frame]
    smoke_pos_current = animation_data['smoke_positions'][frame]
    smoke_ball_pos_current = animation_data['smoke_ball_positions'][frame]
    is_blocked = animation_data['blocking_status'][frame]
    
    # 遮挡状态检测和记录
    if is_blocked and not blocking_detected:
        blocking_start_time = current_time
        blocking_detected = True
        print(f"🚫 遮挡开始时间: {blocking_start_time:.1f}s")
    elif not is_blocked and blocking_detected and blocking_end_time is None:
        blocking_end_time = current_time
        print(f"✅ 遮挡结束时间: {blocking_end_time:.1f}s")
        print(f"⏱️ 遮挡持续时间: {blocking_end_time - blocking_start_time:.1f}s")
        animation_should_stop = True
        # 停止动画
        import matplotlib.pyplot as plt
        plt.close('all')
        return []
    
    # 绘制当前位置的对象
    draw_missile_3d(ax, missile_pos_current, colors['missile'])
    draw_drone_3d(ax, drone_pos_current, colors['drone'])
    
    # 绘制烟雾弹（如果还在飞行中）
    if current_time > flight_time and current_time <= total_time:
        ax.scatter(*smoke_pos_current, s=200, c=colors['smoke'], marker='o', 
                  edgecolors=colors['smoke_glow'], linewidth=2, alpha=1.0)
    
    # 绘制烟雾球（如果已爆炸）
    if smoke_ball_pos_current is not None:
        draw_smoke_ball(ax, smoke_ball_pos_current, smoke_ball_radius, colors['smoke_ball'])
    
    # 绘制轨迹（渐变效果）
    trail_length = min(frame, 60)  # 轨迹长度
    if trail_length > 1:
        # 导弹轨迹
        missile_trail = np.array(animation_data['missile_positions'][max(0, frame-trail_length):frame+1])
        if len(missile_trail) > 1:
            ax.plot(missile_trail[:, 0], missile_trail[:, 1], missile_trail[:, 2], 
                   color=colors['missile'], linewidth=3, alpha=0.7)
        
        # 无人机轨迹
        drone_trail = np.array(animation_data['drone_positions'][max(0, frame-trail_length):frame+1])
        if len(drone_trail) > 1:
            ax.plot(drone_trail[:, 0], drone_trail[:, 1], drone_trail[:, 2], 
                   color=colors['drone'], linewidth=3, alpha=0.7)
    
    # 添加时间显示
    time_text = f'时间: {current_time:.1f}s'
    if current_time <= flight_time:
        phase_text = '阶段: 无人机飞行'
    elif current_time <= total_time:
        phase_text = '阶段: 烟雾弹飞行'
    else:
        phase_text = '阶段: 烟雾遮挡'
    
    # 添加遮挡状态显示
    blocking_text = '🚫 目标被遮挡' if is_blocked else '👁️ 目标可见'
    
    ax.text2D(0.02, 0.98, f'{time_text}\n{phase_text}\n{blocking_text}', transform=ax.transAxes, 
             fontsize=12, fontweight='bold', color=colors['text'],
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
             verticalalignment='top')
    
    # 添加关键事件标记
    events_text = []
    if current_time >= flight_time:
        events_text.append(f'✓ 投弹 ({flight_time}s)')
    if current_time >= total_time:
        events_text.append(f'✓ 爆炸 ({total_time}s)')
    if blocking_start_time is not None:
        events_text.append(f'🚫 遮挡开始 ({blocking_start_time:.1f}s)')
    if blocking_end_time is not None:
        events_text.append(f'✅ 遮挡结束 ({blocking_end_time:.1f}s)')
    
    if events_text:
        ax.text2D(0.02, 0.80, '\n'.join(events_text), transform=ax.transAxes, 
                 fontsize=10, color=colors['accent'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                 verticalalignment='top')
    
    # 如果遮挡结束，停止动画
    if animation_should_stop:
        anim.pause()
        print("🎬 动画已停止 - 遮挡结束")
    
    return []

# 创建动画
print("正在创建动画...")
init_plot()
anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                              interval=1000/fps, blit=False, repeat=True)

# 添加控制按钮
def pause_resume(event):
    if anim.running:
        anim.pause()
    else:
        anim.resume()

# 暂停/播放按钮
ax_pause = plt.axes([0.02, 0.02, 0.08, 0.04])
ax_pause.patch.set_facecolor('black')
button_pause = Button(ax_pause, '暂停/播放', color='gray', hovercolor='lightgray')
button_pause.label.set_color('white')
button_pause.on_clicked(pause_resume)

# 重置动画按钮
def reset_animation(event):
    anim.frame_seq = anim.new_frame_seq()
    anim.event_source.stop()
    anim.event_source.start()

ax_reset = plt.axes([0.11, 0.02, 0.08, 0.04])
ax_reset.patch.set_facecolor('black')
button_reset = Button(ax_reset, '重置动画', color='gray', hovercolor='lightgray')
button_reset.label.set_color('white')
button_reset.on_clicked(reset_animation)

# 保存动画按钮
def save_animation(event):
    print("正在保存动画为GIF...")
    anim.save('missile_drone_animation.gif', writer='pillow', fps=15)
    print("动画已保存为 missile_drone_animation.gif")

ax_save = plt.axes([0.20, 0.02, 0.08, 0.04])
ax_save.patch.set_facecolor('black')
button_save = Button(ax_save, '保存GIF', color='gray', hovercolor='lightgray')
button_save.label.set_color('white')
button_save.on_clicked(save_animation)

print("\n=== 动画控制说明 ===")
print("🎬 暂停/播放: 点击暂停/播放按钮")
print("🔄 重置动画: 点击重置动画按钮")
print("💾 保存GIF: 点击保存GIF按钮")
print("🖱️ 鼠标操作: 可以旋转、缩放视角")
print("⌨️ 键盘操作: p键平移，o键缩放，r键重置视图")
print("==================\n")

plt.tight_layout()
plt.show()

print("动画播放完成！")