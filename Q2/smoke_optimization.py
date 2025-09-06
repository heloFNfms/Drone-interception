#!/usr/bin/env python3
"""
烟幕遮蔽优化程序 - 完整版
专为云GPU环境优化的高性能计算程序
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, dual_annealing
import pandas as pd
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 为无GUI环境设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

class CompleteSmokeOptimizer:
    def __init__(self, use_multiprocessing=True):
        """完整烟幕遮蔽优化器"""
        # 物理常数
        self.g = 9.8  # m/s²
        
        # 初始位置
        self.M0 = np.array([20000, 0, 2000])  # 导弹
        self.D0 = np.array([17800, 0, 1800])  # 无人机
        
        # 运动参数
        self.v_missile = 300  # m/s
        self.smoke_radius = 10  # m
        self.smoke_sink_speed = 3  # m/s
        
        # 目标圆柱体
        self.B_radius = 7  # m
        self.B_height = 10  # m
        self.B_center = np.array([0, 200, 5])
        
        # 约束条件
        self.v_drone_min = 70   # m/s
        self.v_drone_max = 140  # m/s
        
        # 计算设置
        self.use_multiprocessing = use_multiprocessing
        
        print(f"优化器初始化完成")
        print(f"导弹起点: {self.M0}")
        print(f"无人机起点: {self.D0}")
        print(f"目标位置: {self.B_center}")
    
    def missile_position(self, t):
        """计算时间t时导弹位置"""
        direction = -self.M0 / np.linalg.norm(self.M0)
        return self.M0 + direction * self.v_missile * t
    
    def drone_position(self, t, v_drone, direction_angle, throw_time):
        """计算时间t时无人机位置"""
        direction = np.array([np.cos(direction_angle), np.sin(direction_angle), 0])
        actual_time = min(t, throw_time)
        return self.D0 + direction * v_drone * actual_time
    
    def smoke_position(self, t, v_drone, direction_angle, throw_time, boom_time):
        """计算时间t时烟雾位置"""
        if t < throw_time:
            return None
        
        # 投放位置和初速度
        throw_direction = np.array([np.cos(direction_angle), np.sin(direction_angle), 0])
        throw_pos = self.D0 + throw_direction * v_drone * throw_time
        initial_velocity = throw_direction * v_drone
        
        if t < boom_time:
            # 平抛阶段
            dt = t - throw_time
            x = throw_pos[0] + initial_velocity[0] * dt
            y = throw_pos[1] + initial_velocity[1] * dt
            z = throw_pos[2] - 0.5 * self.g * dt**2
            return np.array([x, y, z])
        else:
            # 烟球下沉阶段
            fall_time = boom_time - throw_time
            boom_x = throw_pos[0] + initial_velocity[0] * fall_time
            boom_y = throw_pos[1] + initial_velocity[1] * fall_time
            boom_z = throw_pos[2] - 0.5 * self.g * fall_time**2
            
            sink_time = t - boom_time
            return np.array([boom_x, boom_y, boom_z - self.smoke_sink_speed * sink_time])
    
    def is_point_in_shadow_cone(self, point, missile_pos, smoke_pos):
        """判断点是否在阴影圆锥内"""
        MS = smoke_pos - missile_pos
        dist_MS = np.linalg.norm(MS)
        
        if dist_MS <= self.smoke_radius:
            return False
        
        # 圆锥参数
        sin_alpha = self.smoke_radius / dist_MS
        cos_alpha = np.sqrt(1 - sin_alpha**2)
        tan_alpha = sin_alpha / cos_alpha
        
        cone_axis = MS / dist_MS
        
        # 点到导弹的向量
        MP = point - missile_pos
        proj_length = np.dot(MP, cone_axis)
        
        # 点必须在烟雾球后方
        if proj_length <= dist_MS:
            return False
        
        # 计算垂直距离
        projection = cone_axis * proj_length
        perpendicular = MP - projection
        perp_distance = np.linalg.norm(perpendicular)
        
        # 该位置圆锥半径
        distance_from_smoke = proj_length - dist_MS
        cone_radius_at_point = distance_from_smoke * tan_alpha
        
        return perp_distance <= cone_radius_at_point + 0.05  # 小容差
    
    def calculate_cylinder_shadow_coverage(self, missile_pos, smoke_pos):
        """计算圆柱体被遮蔽的比例"""
        # 高密度采样圆柱体
        sample_points = []
        
        # 轴线采样
        for z in np.linspace(self.B_center[2] - self.B_height/2, 
                           self.B_center[2] + self.B_height/2, 40):
            sample_points.append([self.B_center[0], self.B_center[1], z])
        
        # 表面采样
        for h in np.linspace(0, 1, 20):
            z = self.B_center[2] - self.B_height/2 + h * self.B_height
            for angle in np.linspace(0, 2*np.pi, 48, endpoint=False):
                x = self.B_center[0] + self.B_radius * np.cos(angle)
                y = self.B_center[1] + self.B_radius * np.sin(angle)
                sample_points.append([x, y, z])
        
        # 内部采样
        for h in np.linspace(0.2, 0.8, 8):
            z = self.B_center[2] - self.B_height/2 + h * self.B_height
            for r in np.linspace(0.3, 0.9, 4):
                radius = r * self.B_radius
                for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
                    x = self.B_center[0] + radius * np.cos(angle)
                    y = self.B_center[1] + radius * np.sin(angle)
                    sample_points.append([x, y, z])
        
        # 检查遮蔽
        shielded_points = 0
        total_points = len(sample_points)
        
        for point in sample_points:
            if self.is_point_in_shadow_cone(np.array(point), missile_pos, smoke_pos):
                shielded_points += 1
        
        return shielded_points / total_points
    
    def calculate_shielding_time(self, params):
        """计算总遮蔽时间"""
        v_drone, direction_angle, throw_time, boom_time = params
        
        # 参数有效性检查
        if not (self.v_drone_min <= v_drone <= self.v_drone_max):
            return 0
        if throw_time <= 0 or boom_time <= throw_time:
            return 0
        if boom_time > 20:  # 避免过长计算
            return 0
        
        # 时间步进参数
        dt = 0.008  # 8ms步长
        max_time = min(30, boom_time + 20)
        
        total_shielding_time = 0
        currently_shielded = False
        shield_start_time = None
        
        t = boom_time  # 从烟球形成开始
        
        # 预计算导弹到达目标的时间
        missile_flight_time = np.linalg.norm(self.M0) / self.v_missile
        
        while t <= max_time and t < missile_flight_time * 0.98:
            missile_pos = self.missile_position(t)
            smoke_pos = self.smoke_position(t, v_drone, direction_angle, throw_time, boom_time)
            
            if smoke_pos is None or smoke_pos[2] < -200:  # 烟球过低
                break
            
            # 快速距离检查
            if np.linalg.norm(smoke_pos - missile_pos) > 2000:
                t += dt
                continue
            
            # 计算遮蔽比例
            coverage = self.calculate_cylinder_shadow_coverage(missile_pos, smoke_pos)
            is_shielded = coverage > 0.12  # 遮蔽阈值
            
            if is_shielded and not currently_shielded:
                # 开始遮蔽
                currently_shielded = True
                shield_start_time = t
            elif not is_shielded and currently_shielded:
                # 结束遮蔽
                currently_shielded = False
                if shield_start_time is not None:
                    total_shielding_time += t - shield_start_time
                # 假设只有一个连续遮蔽期
                break
            
            t += dt
        
        # 如果仿真结束时仍在遮蔽状态
        if currently_shielded and shield_start_time is not None:
            total_shielding_time += t - shield_start_time
        
        return total_shielding_time
    
    def objective_function(self, params):
        """优化目标函数（负遮蔽时间）"""
        return -self.calculate_shielding_time(params)
    
    def optimize_with_multiple_strategies(self):
        """多策略优化"""
        print("\n=== 开始多策略优化 ===")
        
        # 参数边界
        bounds = [
            (self.v_drone_min, self.v_drone_max),  # 速度
            (0, 2*np.pi),                          # 方向角
            (0.5, 6.0),                            # 投放时间  
            (1.0, 12.0)                            # 爆炸时间
        ]
        
        results = {}
        
        # 策略1: 差分进化
        print("执行差分进化算法...")
        start_time = time.time()
        
        result_de = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=300,
            popsize=20,
            seed=42,
            workers=1,  # 避免并行问题
            polish=True
        )
        
        time_de = time.time() - start_time
        results['differential_evolution'] = {
            'params': result_de.x,
            'shielding_time': -result_de.fun,
            'computation_time': time_de,
            'success': result_de.success
        }
        
        print(f"差分进化完成: {time_de:.1f}s, 遮蔽时间: {-result_de.fun:.6f}s")
        
        # 策略2: 双重退火
        print("执行双重退火算法...")
        start_time = time.time()
        
        result_da = dual_annealing(
            self.objective_function,
            bounds,
            maxiter=500,
            seed=42
        )
        
        time_da = time.time() - start_time
        results['dual_annealing'] = {
            'params': result_da.x,
            'shielding_time': -result_da.fun,
            'computation_time': time_da,
            'success': result_da.success
        }
        
        print(f"双重退火完成: {time_da:.1f}s, 遮蔽时间: {-result_da.fun:.6f}s")
        
        # 策略3: 基于最好结果的局部优化
        print("执行局部精细优化...")
        start_time = time.time()
        
        best_so_far = max(results.values(), key=lambda x: x['shielding_time'])
        
        result_local = minimize(
            self.objective_function,
            best_so_far['params'],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        time_local = time.time() - start_time
        results['local_search'] = {
            'params': result_local.x,
            'shielding_time': -result_local.fun,
            'computation_time': time_local,
            'success': result_local.success
        }
        
        print(f"局部优化完成: {time_local:.1f}s, 遮蔽时间: {-result_local.fun:.6f}s")
        
        # 选择最佳结果
        best_method = max(results.keys(), key=lambda k: results[k]['shielding_time'])
        best_result = results[best_method]
        
        total_time = sum(r['computation_time'] for r in results.values())
        
        print(f"\n最佳策略: {best_method}")
        print(f"总计算时间: {total_time:.1f}s")
        print(f"最大遮蔽时间: {best_result['shielding_time']:.6f}s")
        
        return best_result['params'], best_result['shielding_time'], best_method, results
    
    def analyze_optimal_solution(self, params):
        """分析最优解"""
        v_drone, direction_angle, throw_time, boom_time = params
        
        print(f"\n=== 最优解详细分析 ===")
        print(f"无人机速度: {v_drone:.2f} m/s")
        print(f"飞行方向: {np.degrees(direction_angle):.1f}° ({direction_angle:.4f} rad)")
        print(f"投放时间: {throw_time:.3f} s")
        print(f"爆炸时间: {boom_time:.3f} s")
        print(f"平抛持续: {boom_time - throw_time:.3f} s")
        
        # 关键位置计算
        throw_pos = self.drone_position(throw_time, v_drone, direction_angle, throw_time)
        boom_pos = self.smoke_position(boom_time, v_drone, direction_angle, throw_time, boom_time)
        missile_at_boom = self.missile_position(boom_time)
        
        print(f"\n关键位置:")
        print(f"投放点: ({throw_pos[0]:.0f}, {throw_pos[1]:.0f}, {throw_pos[2]:.0f})")
        print(f"爆炸点: ({boom_pos[0]:.0f}, {boom_pos[1]:.0f}, {boom_pos[2]:.0f})")
        print(f"爆炸时导弹位置: ({missile_at_boom[0]:.0f}, {missile_at_boom[1]:.0f}, {missile_at_boom[2]:.0f})")
        
        # 几何分析
        throw_to_target = np.linalg.norm(throw_pos[:2] - self.B_center[:2])
        boom_to_target = np.linalg.norm(boom_pos[:2] - self.B_center[:2])
        
        print(f"\n几何分析:")
        print(f"投放点到目标水平距离: {throw_to_target:.0f} m")
        print(f"爆炸点到目标水平距离: {boom_to_target:.0f} m")
        print(f"爆炸高度: {boom_pos[2]:.0f} m")
        
        # 重新计算精确遮蔽时间
        precise_time = self.calculate_shielding_time(params)
        print(f"\n遮蔽性能: {precise_time:.6f} s")
        
        return precise_time
    
    def sensitivity_analysis(self, params):
        """敏感性分析"""
        print(f"\n=== 敏感性分析 ===")
        
        param_names = ['速度 (m/s)', '方向角 (rad)', '投放时间 (s)', '爆炸时间 (s)']
        base_time = self.calculate_shielding_time(params)
        
        print(f"基准遮蔽时间: {base_time:.6f} s")
        
        for i, (name, value) in enumerate(zip(param_names, params)):
            # 计算数值导数
            delta = value * 0.01 if value != 0 else 0.01
            
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += delta
            params_minus[i] -= delta
            
            time_plus = self.calculate_shielding_time(params_plus)
            time_minus = self.calculate_shielding_time(params_minus)
            
            sensitivity = (time_plus - time_minus) / (2 * delta)
            relative_sens = sensitivity * value / base_time if base_time > 0 else 0
            
            print(f"{name:15}: {sensitivity:8.5f} (相对: {relative_sens:6.3f})")
    
    def create_visualization(self, params, save_path="optimization_visualization.png"):
        """生成可视化图表"""
        print(f"\n生成可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('烟幕遮蔽优化结果', fontsize=16, fontweight='bold')
        
        v_drone, direction_angle, throw_time, boom_time = params
        
        # 1. 3D轨迹图
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        
        # 导弹轨迹
        t_missile = np.linspace(0, 20, 100)
        missile_traj = np.array([self.missile_position(t) for t in t_missile])
        ax.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 
               'r-', label='导弹轨迹', linewidth=2)
        
        # 无人机轨迹
        t_drone = np.linspace(0, throw_time, 50)
        drone_traj = np.array([self.drone_position(t, v_drone, direction_angle, throw_time) 
                              for t in t_drone])
        ax.plot(drone_traj[:, 0], drone_traj[:, 1], drone_traj[:, 2], 
               'b-', label='无人机轨迹', linewidth=2)
        
        # 烟雾轨迹
        t_smoke = np.linspace(throw_time, boom_time + 8, 80)
        smoke_traj = []
        for t in t_smoke:
            pos = self.smoke_position(t, v_drone, direction_angle, throw_time, boom_time)
            if pos is not None and pos[2] > -100:
                smoke_traj.append(pos)
        
        if smoke_traj:
            smoke_traj = np.array(smoke_traj)
            ax.plot(smoke_traj[:, 0], smoke_traj[:, 1], smoke_traj[:, 2], 
                   'gray', alpha=0.7, label='烟雾轨迹')
        
        # 标记关键点
        ax.scatter(*self.M0, color='red', s=100, marker='^')
        ax.scatter(*self.D0, color='blue', s=100, marker='s')
        ax.scatter(*self.B_center, color='green', s=200, marker='o')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title('三维轨迹图')
        
        # 2. 参数敏感性
        speeds = np.linspace(70, 140, 15)
        shield_times = []
        for speed in speeds:
            test_params = params.copy()
            test_params[0] = speed
            shield_times.append(self.calculate_shielding_time(test_params))
        
        axes[0,1].plot(speeds, shield_times, 'b-o', markersize=4)
        axes[0,1].axvline(params[0], color='red', linestyle='--', 
                         label=f'最优: {params[0]:.1f}')
        axes[0,1].set_xlabel('速度 (m/s)')
        axes[0,1].set_ylabel('遮蔽时间 (s)')
        axes[0,1].set_title('速度敏感性')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # 3. 方向角敏感性  
        angles = np.linspace(0, 2*np.pi, 24)
        angle_shield_times = []
        for angle in angles:
            test_params = params.copy()
            test_params[1] = angle
            angle_shield_times.append(self.calculate_shielding_time(test_params))
        
        axes[1,0].plot(np.degrees(angles), angle_shield_times, 'g-o', markersize=4)
        axes[1,0].axvline(np.degrees(params[1]), color='red', linestyle='--',
                         label=f'最优: {np.degrees(params[1]):.1f}°')
        axes[1,0].set_xlabel('方向角 (°)')
        axes[1,0].set_ylabel('遮蔽时间 (s)')
        axes[1,0].set_title('方向角敏感性')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # 4. 时间参数热图
        throw_range = np.linspace(0.5, 4, 12)
        boom_range = np.linspace(2, 10, 12)
        T, B = np.meshgrid(throw_range, boom_range)
        Z = np.zeros_like(T)
        
        for i, tt in enumerate(throw_range):
            for j, bb in enumerate(boom_range):
                if bb > tt:
                    test_params = params.copy()
                    test_params[2] = tt
                    test_params[3] = bb
                    Z[j, i] = self.calculate_shielding_time(test_params)
        
        im = axes[1,1].contourf(T, B, Z, levels=15, cmap='viridis')
        axes[1,1].scatter(params[2], params[3], color='red', s=100, 
                         marker='*', label='最优点')
        axes[1,1].set_xlabel('投放时间 (s)')
        axes[1,1].set_ylabel('爆炸时间 (s)')
        axes[1,1].set_title('时间参数优化空间')
        axes[1,1].legend()
        plt.colorbar(im, ax=axes[1,1], label='遮蔽时间 (s)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存: {save_path}")
    
    def generate_report(self, params, shielding_time, method, results, 
                       save_path="optimization_report.txt"):
        """生成详细报告"""
        print(f"\n生成详细报告...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("烟幕遮蔽优化分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最优算法: {method}\n")
            f.write(f"最大遮蔽时间: {shielding_time:.6f} 秒\n\n")
            
            f.write("=" * 40 + "\n")
            f.write("最优参数配置\n")
            f.write("=" * 40 + "\n")
            f.write(f"无人机速度: {params[0]:.3f} m/s\n")
            f.write(f"飞行方向: {np.degrees(params[1]):.2f}° ({params[1]:.4f} rad)\n")
            f.write(f"投放时间: {params[2]:.3f} s\n")
            f.write(f"爆炸时间: {params[3]:.3f} s\n")
            f.write(f"平抛持续: {params[3] - params[2]:.3f} s\n\n")
            
            # 关键位置信息
            throw_pos = self.drone_position(params[2], params[0], params[1], params[2])
            boom_pos = self.smoke_position(params[3], params[0], params[1], params[2], params[3])
            
            f.write("=" * 40 + "\n")
            f.write("关键位置分析\n")
            f.write("=" * 40 + "\n")
            f.write(f"投放点坐标: ({throw_pos[0]:.1f}, {throw_pos[1]:.1f}, {throw_pos[2]:.1f})\n")
            f.write(f"爆炸点坐标: ({boom_pos[0]:.1f}, {boom_pos[1]:.1f}, {boom_pos[2]:.1f})\n")
            f.write(f"投放点到目标距离: {np.linalg.norm(throw_pos[:2] - self.B_center[:2]):.1f} m\n")
            f.write(f"爆炸点到目标距离: {np.linalg.norm(boom_pos[:2] - self.B_center[:2]):.1f} m\n\n")
            
            # 算法性能对比
            f.write("=" * 40 + "\n")
            f.write("算法性能对比\n")
            f.write("=" * 40 + "\n")
            for alg_name, result in results.items():
                f.write(f"{alg_name:20}: {result['shielding_time']:.6f}s ({result['computation_time']:.1f}s)\n")
            
        print(f"报告已保存: {save_path}")


def main():
    """主程序入口"""
    print("=" * 60)
    print("烟幕遮蔽优化系统 - 完整版")
    print("=" * 60)
    
    # 环境检查
    try:
        import psutil
        cpu_count = os.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"系统资源: {cpu_count}核CPU, {memory_gb:.1f}GB内存")
    except ImportError:
        print("系统资源检查跳过 (psutil未安装)")
    
    # 初始化优化器
    print("\n初始化优化器...")
    optimizer = CompleteSmokeOptimizer(use_multiprocessing=True)
    
    try:
        # 执行优化
        start_time = time.time()
        optimal_params, max_shielding_time, best_method, all_results = optimizer.optimize_with_multiple_strategies()
        total_optimization_time = time.time() - start_time
        
        # 详细分析
        print("\n" + "=" * 60)
        print("优化结果")
        print("=" * 60)
        
        actual_time = optimizer.analyze_optimal_solution(optimal_params)
        optimizer.sensitivity_analysis(optimal_params)
        
        # 与原始设计对比
        print("\n" + "=" * 60)
        print("性能对比")
        print("=" * 60)
        
        # 原始参数（基于代码推断）
        original_direction = np.arctan2(-optimizer.D0[1], -optimizer.D0[0])  # 朝向原点
        original_params = [120.0, original_direction, 1.5, 5.1]  # 原代码参数
        original_time = optimizer.calculate_shielding_time(original_params)
        
        print(f"原始设计遮蔽时间: {original_time:.6f} 秒")
        print(f"优化后遮蔽时间: {max_shielding_time:.6f} 秒")
        
        if original_time > 0:
            improvement_ratio = max_shielding_time / original_time
            print(f"性能提升: {improvement_ratio:.2f}倍")
            print(f"绝对增益: +{max_shielding_time - original_time:.6f} 秒")
        else:
            print("原始设计无遮蔽效果，优化后实现遮蔽")
        
        print(f"计算效率: {max_shielding_time/total_optimization_time*1000:.2f} ms遮蔽时间/计算秒")
        
        # 生成可视化和报告
        optimizer.create_visualization(optimal_params)
        optimizer.generate_report(optimal_params, max_shielding_time, best_method, all_results)
        
        # 保存JSON结果
        final_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimal_parameters': {
                'drone_speed': float(optimal_params[0]),
                'flight_direction_deg': float(np.degrees(optimal_params[1])),
                'flight_direction_rad': float(optimal_params[1]),
                'throw_time': float(optimal_params[2]),
                'boom_time': float(optimal_params[3]),
                'fall_duration': float(optimal_params[3] - optimal_params[2])
            },
            'performance': {
                'max_shielding_time': float(max_shielding_time),
                'original_shielding_time': float(original_time),
                'improvement_ratio': float(max_shielding_time / original_time) if original_time > 0 else None,
                'absolute_improvement': float(max_shielding_time - original_time)
            },
            'optimization': {
                'best_method': best_method,
                'total_computation_time': float(total_optimization_time),
                'algorithm_results': {k: {
                    'shielding_time': float(v['shielding_time']),
                    'computation_time': float(v['computation_time']),
                    'success': bool(v['success'])
                } for k, v in all_results.items()}
            },
            'key_positions': {
                'throw_position': optimizer.drone_position(optimal_params[2], optimal_params[0], optimal_params[1], optimal_params[2]).tolist(),
                'boom_position': optimizer.smoke_position(optimal_params[3], optimal_params[0], optimal_params[1], optimal_params[2], optimal_params[3]).tolist()
            }
        }
        
        with open('optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nJSON结果已保存: optimization_results.json")
        
        # 最终总结
        print("\n" + "=" * 60)
        print("优化完成总结")
        print("=" * 60)
        print(f"最优遮蔽时间: {max_shielding_time:.6f} 秒")
        print(f"最优无人机速度: {optimal_params[0]:.1f} m/s")
        print(f"最优飞行方向: {np.degrees(optimal_params[1]):.1f}°")
        print(f"最优投放时机: {optimal_params[2]:.2f} 秒")
        print(f"最优爆炸时机: {optimal_params[3]:.2f} 秒")
        print(f"总计算时间: {total_optimization_time:.1f} 秒")
        print(f"使用算法: {best_method}")
        
        # 实用建议
        print(f"\n实用建议:")
        print(f"1. 无人机应以 {optimal_params[0]:.0f} m/s 的速度飞行")
        print(f"2. 飞行方向应为 {np.degrees(optimal_params[1]):.0f}° (从正X轴逆时针测量)")
        print(f"3. 在 {optimal_params[2]:.2f} 秒时投放烟雾弹")
        print(f"4. 烟雾弹应在 {optimal_params[3]:.2f} 秒时爆炸")
        print(f"5. 预期可获得 {max_shielding_time:.3f} 秒的有效遮蔽时间")
        
    except KeyboardInterrupt:
        print("\n用户中断计算")
    except Exception as e:
        print(f"\n计算过程发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n程序执行结束")


if __name__ == "__main__":
    # 设置随机种子确保结果可重现
    np.random.seed(42)
    
    # 忽略警告信息
    import warnings
    warnings.filterwarnings('ignore')
    
    # 执行主程序
    main()