import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

class SmokeShieldingOptimizer:
    def __init__(self):
        # 固定参数
        self.g = 9.8  # 重力加速度 m/s²
        self.M0 = np.array([20000, 0, 2000])  # 导弹初始位置
        self.D0 = np.array([17800, 0, 1800])  # 无人机初始位置
        self.v_missile = 300  # 导弹速度 m/s
        self.smoke_radius = 10  # 烟球半径 m
        self.smoke_sink_speed = 3  # 烟球下沉速度 m/s
        
        # 目标圆柱体参数
        self.B_radius = 7  # m
        self.B_height = 10  # m  
        self.B_center = np.array([0, 200, 5])  # 圆柱中心
        
        # 速度约束
        self.v_drone_min = 70  # m/s
        self.v_drone_max = 140  # m/s
    
    def missile_position(self, t):
        """导弹在时间t的位置"""
        direction = -self.M0 / np.linalg.norm(self.M0)  # 指向原点的单位向量
        return self.M0 + direction * self.v_missile * t
    
    def drone_position(self, t, v_drone, direction_angle, throw_time):
        """无人机在时间t的位置"""
        # direction_angle为水平面内的飞行方向角度（弧度）
        direction = np.array([np.cos(direction_angle), np.sin(direction_angle), 0])
        
        if t <= throw_time:
            return self.D0 + direction * v_drone * t
        else:
            return self.D0 + direction * v_drone * throw_time
    
    def smoke_position(self, t, v_drone, direction_angle, throw_time, boom_time):
        """烟雾弹/烟球在时间t的位置"""
        if t < throw_time:
            return None  # 未投放
        
        # 投放点
        throw_direction = np.array([np.cos(direction_angle), np.sin(direction_angle), 0])
        throw_pos = self.D0 + throw_direction * v_drone * throw_time
        
        if t < boom_time:
            # 平抛阶段
            dt = t - throw_time
            # 水平初速度等于无人机速度
            initial_velocity = throw_direction * v_drone
            x = throw_pos[0] + initial_velocity[0] * dt
            y = throw_pos[1] + initial_velocity[1] * dt
            z = throw_pos[2] - 0.5 * self.g * dt**2
            return np.array([x, y, z])
        else:
            # 烟球下沉阶段
            fall_time = boom_time - throw_time
            # 爆炸点位置
            initial_velocity = throw_direction * v_drone
            boom_pos = np.array([
                throw_pos[0] + initial_velocity[0] * fall_time,
                throw_pos[1] + initial_velocity[1] * fall_time, 
                throw_pos[2] - 0.5 * self.g * fall_time**2
            ])
            
            # 下沉时间
            sink_time = t - boom_time
            return boom_pos - np.array([0, 0, self.smoke_sink_speed * sink_time])
    
    def is_cylinder_in_shadow_cone(self, missile_pos, smoke_pos):
        """判断圆柱体是否被阴影圆锥遮蔽"""
        MS = smoke_pos - missile_pos
        dist_MS = np.linalg.norm(MS)
        
        if dist_MS <= self.smoke_radius:
            return False
        
        # 切线圆锥参数
        sin_alpha = self.smoke_radius / dist_MS
        cos_alpha = np.sqrt(1 - sin_alpha**2)
        tan_alpha = sin_alpha / cos_alpha
        
        cone_axis = MS / dist_MS
        
        # 高密度采样圆柱体
        sampling_points = []
        
        # 轴线采样
        for h in np.linspace(0, 1, 50):
            z = self.B_center[2] - self.B_height/2 + h * self.B_height
            sampling_points.append([self.B_center[0], self.B_center[1], z])
        
        # 表面采样
        for h in np.linspace(0, 1, 20):
            z = self.B_center[2] - self.B_height/2 + h * self.B_height
            for angle in np.linspace(0, 2*np.pi, 48, endpoint=False):
                x = self.B_center[0] + self.B_radius * np.cos(angle)
                y = self.B_center[1] + self.B_radius * np.sin(angle)
                sampling_points.append([x, y, z])
        
        # 内部采样
        for h in np.linspace(0.1, 0.9, 10):
            z = self.B_center[2] - self.B_height/2 + h * self.B_height
            for r in np.linspace(0.2, 0.8, 5):
                radius = r * self.B_radius
                for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
                    x = self.B_center[0] + radius * np.cos(angle)
                    y = self.B_center[1] + radius * np.sin(angle)
                    sampling_points.append([x, y, z])
        
        points_in_shadow = 0
        total_points = len(sampling_points)
        
        for point in sampling_points:
            point = np.array(point)
            MP = point - missile_pos
            proj_length = np.dot(MP, cone_axis)
            
            if proj_length > dist_MS:
                projection = cone_axis * proj_length
                perpendicular = MP - projection
                perp_distance = np.linalg.norm(perpendicular)
                
                distance_from_smoke = proj_length - dist_MS
                cone_radius_at_point = distance_from_smoke * tan_alpha
                
                if perp_distance <= cone_radius_at_point + 0.1:  # 小容差
                    points_in_shadow += 1
        
        shadow_ratio = points_in_shadow / total_points
        return shadow_ratio > 0.12  # 遮蔽阈值
    
    def calculate_shielding_time(self, params):
        """计算给定参数下的总遮蔽时间"""
        v_drone, direction_angle, throw_time, boom_time = params
        
        # 参数有效性检查
        if not (self.v_drone_min <= v_drone <= self.v_drone_max):
            return 0
        if throw_time <= 0 or boom_time <= throw_time:
            return 0
        
        # 时间步长
        dt = 0.01
        max_time = 30.0  # 最大仿真时间
        
        shielded = False
        shield_start_time = None
        shield_end_time = None
        total_shield_time = 0
        
        t = boom_time  # 从烟球形成开始检查
        
        while t <= max_time:
            missile_pos = self.missile_position(t)
            smoke_pos = self.smoke_position(t, v_drone, direction_angle, throw_time, boom_time)
            
            if smoke_pos is None:
                t += dt
                continue
            
            # 检查导弹是否还在飞行中
            if np.linalg.norm(missile_pos) < 100:  # 导弹接近目标
                break
            
            currently_shielded = self.is_cylinder_in_shadow_cone(missile_pos, smoke_pos)
            
            if currently_shielded and not shielded:
                # 开始遮蔽
                shielded = True
                shield_start_time = t
            elif not currently_shielded and shielded:
                # 结束遮蔽
                shielded = False
                shield_end_time = t
                total_shield_time += shield_end_time - shield_start_time
                break  # 假设只有一次连续遮蔽期
            
            t += dt
        
        # 如果仿真结束时仍在遮蔽状态
        if shielded and shield_start_time is not None:
            total_shield_time += t - shield_start_time
        
        return total_shield_time
    
    def objective_function(self, params):
        """优化目标函数：最大化遮蔽时间"""
        shielding_time = self.calculate_shielding_time(params)
        return -shielding_time  # 最小化负值 = 最大化正值
    
    def optimize_parameters(self):
        """优化参数以获得最大遮蔽时间"""
        print("开始优化分析...")
        
        # 参数边界
        # [v_drone, direction_angle, throw_time, boom_time]
        bounds = [
            (self.v_drone_min, self.v_drone_max),  # 无人机速度
            (0, 2*np.pi),  # 飞行方向角度
            (0.5, 5.0),    # 投放时间
            (1.0, 10.0)    # 爆炸时间
        ]
        
        # 使用差分进化算法进行全局优化
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=300,
            popsize=20,
            seed=42,
            polish=True,
            atol=1e-6
        )
        
        optimal_params = result.x
        max_shielding_time = -result.fun
        
        return optimal_params, max_shielding_time
    
    def analyze_sensitivity(self, optimal_params):
        """敏感性分析"""
        print("\n进行敏感性分析...")
        
        base_time = self.calculate_shielding_time(optimal_params)
        sensitivities = []
        param_names = ['速度 (m/s)', '方向角 (rad)', '投放时间 (s)', '爆炸时间 (s)']
        
        for i in range(len(optimal_params)):
            delta = optimal_params[i] * 0.01  # 1%扰动
            
            params_plus = optimal_params.copy()
            params_plus[i] += delta
            time_plus = self.calculate_shielding_time(params_plus)
            
            params_minus = optimal_params.copy()
            params_minus[i] -= delta  
            time_minus = self.calculate_shielding_time(params_minus)
            
            sensitivity = (time_plus - time_minus) / (2 * delta)
            sensitivities.append(sensitivity)
        
        print("参数敏感性分析:")
        for i, (name, sens) in enumerate(zip(param_names, sensitivities)):
            print(f"{name}: {sens:.6f} s/单位")
        
        return sensitivities
    
    def detailed_analysis(self, params):
        """详细分析给定参数的遮蔽过程"""
        v_drone, direction_angle, throw_time, boom_time = params
        
        print(f"\n详细分析 - 参数:")
        print(f"无人机速度: {v_drone:.2f} m/s")
        print(f"飞行方向: {direction_angle:.4f} rad ({np.degrees(direction_angle):.2f}°)")
        print(f"投放时间: {throw_time:.3f} s")
        print(f"爆炸时间: {boom_time:.3f} s")
        print(f"平抛时间: {boom_time - throw_time:.3f} s")
        
        # 计算关键位置
        throw_pos = self.drone_position(throw_time, v_drone, direction_angle, throw_time)
        boom_pos = self.smoke_position(boom_time, v_drone, direction_angle, throw_time, boom_time)
        
        print(f"\n关键位置:")
        print(f"投放点: ({throw_pos[0]:.1f}, {throw_pos[1]:.1f}, {throw_pos[2]:.1f})")
        print(f"爆炸点: ({boom_pos[0]:.1f}, {boom_pos[1]:.1f}, {boom_pos[2]:.1f})")
        
        # 分析遮蔽过程
        dt = 0.01
        t = boom_time
        max_time = 20.0
        
        shield_periods = []
        current_period_start = None
        
        while t <= max_time:
            missile_pos = self.missile_position(t)
            smoke_pos = self.smoke_position(t, v_drone, direction_angle, throw_time, boom_time)
            
            if np.linalg.norm(missile_pos) < 100:
                break
                
            is_shielded = self.is_cylinder_in_shadow_cone(missile_pos, smoke_pos)
            
            if is_shielded and current_period_start is None:
                current_period_start = t
            elif not is_shielded and current_period_start is not None:
                shield_periods.append((current_period_start, t))
                current_period_start = None
            
            t += dt
        
        if current_period_start is not None:
            shield_periods.append((current_period_start, t))
        
        total_time = sum(end - start for start, end in shield_periods)
        
        print(f"\n遮蔽分析:")
        print(f"遮蔽段数: {len(shield_periods)}")
        for i, (start, end) in enumerate(shield_periods):
            print(f"第{i+1}段: {start:.3f}s - {end:.3f}s (持续 {end-start:.3f}s)")
        print(f"总遮蔽时间: {total_time:.3f}s")
        
        return total_time, shield_periods

def main():
    """主函数"""
    optimizer = SmokeShieldingOptimizer()
    
    print("三维烟幕遮蔽优化分析")
    print("="*50)
    
    # 执行优化
    optimal_params, max_time = optimizer.optimize_parameters()
    
    print(f"\n优化结果:")
    print(f"最大遮蔽时间: {max_time:.6f} 秒")
    print(f"最优参数:")
    print(f"  无人机速度: {optimal_params[0]:.2f} m/s")
    print(f"  飞行方向: {optimal_params[1]:.4f} rad ({np.degrees(optimal_params[1]):.2f}°)")
    print(f"  投放时间: {optimal_params[2]:.3f} s") 
    print(f"  爆炸时间: {optimal_params[3]:.3f} s")
    
    # 敏感性分析
    optimizer.analyze_sensitivity(optimal_params)
    
    # 详细分析
    optimizer.detailed_analysis(optimal_params)
    
    # 对比分析：原始参数 vs 优化参数
    print("\n" + "="*50)
    print("对比分析:")
    
    # 原始参数（根据代码推断）
    original_direction = np.arctan2(-optimizer.D0[1], -optimizer.D0[0])  # 朝向原点
    original_params = [120, original_direction, 1.5, 5.1]
    original_time = optimizer.calculate_shielding_time(original_params)
    
    print(f"原始设计遮蔽时间: {original_time:.6f} 秒")
    print(f"优化后遮蔽时间: {max_time:.6f} 秒")
    print(f"改进倍数: {max_time/original_time if original_time > 0 else 'infinity':.2f}倍")
    
    return optimal_params, max_time

if __name__ == "__main__":
    optimal_params, max_time = main()