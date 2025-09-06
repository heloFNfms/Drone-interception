#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导弹-烟球-圆柱遮蔽参数优化算法 (交互式版本)
基于超高精度遮蔽判定算法的参数优化与最长遮蔽时间计算
支持实时可视化和参数调整
"""

import numpy as np
import math
import random
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU加速已启用 (CuPy)")
except ImportError:
    print("警告: CuPy未安装，将使用CPU计算")
    import numpy as cp
    GPU_AVAILABLE = False

@dataclass
class OptimizationResult:
    """优化结果数据类"""
    block_time: float
    params: Dict[str, float]
    trajectory_data: Dict
    generation: int
    fitness_history: List[float]

class Vector3GPU:
    """GPU加速的三维向量类"""
    def __init__(self, x, y, z):
        if GPU_AVAILABLE:
            self.data = cp.array([x, y, z], dtype=cp.float32)
        else:
            self.data = np.array([x, y, z], dtype=np.float32)
    
    @property
    def x(self):
        return float(self.data[0])
    
    @property
    def y(self):
        return float(self.data[1])
    
    @property
    def z(self):
        return float(self.data[2])
    
    def __add__(self, other):
        result = Vector3GPU(0, 0, 0)
        result.data = self.data + other.data
        return result
    
    def __sub__(self, other):
        result = Vector3GPU(0, 0, 0)
        result.data = self.data - other.data
        return result
    
    def __mul__(self, scalar):
        result = Vector3GPU(0, 0, 0)
        result.data = self.data * scalar
        return result
    
    def dot(self, other):
        if GPU_AVAILABLE:
            return float(cp.dot(self.data, other.data))
        else:
            return float(np.dot(self.data, other.data))
    
    def length(self):
        if GPU_AVAILABLE:
            return float(cp.linalg.norm(self.data))
        else:
            return float(np.linalg.norm(self.data))
    
    def normalize(self):
        length = self.length()
        if length > 1e-10:
            return self * (1.0 / length)
        return Vector3GPU(1, 0, 0)
    
    def distance_to(self, other):
        return (self - other).length()
    
    def to_cpu(self):
        if GPU_AVAILABLE:
            return Vector3GPU(float(self.data[0].get()), float(self.data[1].get()), float(self.data[2].get()))
        return self
    
    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z}

class OptimizationAlgorithmInteractive:
    """交互式导弹-烟球-圆柱遮蔽参数优化算法"""
    
    def __init__(self):
        # 物理常量和初始条件
        self.g = 9.8  # 重力加速度 m/s²
        self.v_missile = 300  # 导弹速度 m/s
        
        # 导弹初始位置 M(20000,0,2000)
        self.M0 = Vector3GPU(20000, 0, 2000)
        
        # 圆柱体目标参数 B(0,200,0)
        self.B_CENTER = Vector3GPU(0, 200, 0)  # 圆柱体中心（高度中心）
        self.B_RADIUS = 7  # 圆柱体半径 m
        self.B_HEIGHT = 10  # 圆柱体高度 m
        
        # 固定参数
        self.SMOKE_RADIUS = 10  # 烟球半径固定为10m
        self.DRONE_HEIGHT = 1800  # 无人机飞行高度固定为1800m
        
        # 参数范围定义 - 根据a.html的要求修正
        self.param_ranges = {
            'droneSpeed': (70, 140),      # 无人机速度范围 m/s
            'droneDirection': (0, 360),   # 飞行方向角度范围 度
            'throwTime': (0.5, 3.0),      # 投放时间范围 s
            'fallTime': (2.0, 5.0),       # 平抛时间范围 s
        }
        
        # 固定参数（来自a.html）
        self.DRONE_INIT_X = 17800     # 无人机初始位置X（固定）
        self.DRONE_INIT_Y = 0         # 无人机初始位置Y（固定）
        self.SINK_SPEED = 3           # 烟球下沉速度（固定）m/s
        
        # 优化算法参数
        self.population_size = 100 if GPU_AVAILABLE else 50
        self.max_generations = 300 if GPU_AVAILABLE else 150
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8
        self.elite_size = 10 if GPU_AVAILABLE else 5
        
        # 优化状态
        self.best_result = None
        self.optimization_history = []
        self.current_generation = 0
        
        # 预生成采样点
        self._precompute_sampling_points()
    
    def _precompute_sampling_points(self):
        """预计算圆柱体采样点（超高精度）"""
        sampling_points = []
        
        # 1. 圆柱体轴线上的超密集采样
        for h in np.arange(0, 1.01, 0.02):
            z = self.B_CENTER.z - self.B_HEIGHT / 2 + h * self.B_HEIGHT
            sampling_points.append([self.B_CENTER.x, self.B_CENTER.y, z])
        
        # 2. 圆柱体表面的超密集采样
        for h in np.arange(0, 1.01, 0.05):
            z = self.B_CENTER.z - self.B_HEIGHT / 2 + h * self.B_HEIGHT
            for angle in np.arange(0, 2 * np.pi, np.pi / 24):
                x = self.B_CENTER.x + self.B_RADIUS * np.cos(angle)
                y = self.B_CENTER.y + self.B_RADIUS * np.sin(angle)
                sampling_points.append([x, y, z])
        
        # 3. 圆柱体内部的超密集采样
        for h in np.arange(0.1, 0.91, 0.1):
            z = self.B_CENTER.z - self.B_HEIGHT / 2 + h * self.B_HEIGHT
            for r in np.arange(0.1, 0.96, 0.15):
                radius = r * self.B_RADIUS
                for angle in np.arange(0, 2 * np.pi, np.pi / 12):
                    x = self.B_CENTER.x + radius * np.cos(angle)
                    y = self.B_CENTER.y + radius * np.sin(angle)
                    sampling_points.append([x, y, z])
        
        # 4. 边缘特殊采样
        for h in np.arange(0, 1.01, 0.02):
            z = self.B_CENTER.z - self.B_HEIGHT / 2 + h * self.B_HEIGHT
            for angle in np.arange(0, 2 * np.pi, np.pi / 36):
                for r_factor in np.arange(0.95, 1.01, 0.025):
                    radius = r_factor * self.B_RADIUS
                    x = self.B_CENTER.x + radius * np.cos(angle)
                    y = self.B_CENTER.y + radius * np.sin(angle)
                    sampling_points.append([x, y, z])
        
        if GPU_AVAILABLE:
            self.sampling_points = cp.array(sampling_points, dtype=cp.float32)
        else:
            self.sampling_points = np.array(sampling_points, dtype=np.float32)
        
        print(f"预计算采样点数: {len(sampling_points)}")
    
    def get_missile_position(self, t):
        """计算导弹在时刻t的位置"""
        # 导弹朝向原点匀速直线运动
        direction = Vector3GPU(-self.M0.x, -self.M0.y, -self.M0.z).normalize()
        return self.M0 + direction * (self.v_missile * t)
    
    def get_smoke_position(self, t, D0, drone_dir, drone_speed, throw_time, fall_time, sink_speed):
        """计算烟雾在时刻t的位置"""
        if t < throw_time:
            return None  # 还未投放
        
        boom_time = throw_time + fall_time
        
        # 投放点位置
        throw_pos = D0 + drone_dir * (drone_speed * throw_time)
        
        if t < boom_time:
            # 平抛阶段
            t_fall = t - throw_time
            horizontal_vel = drone_dir * drone_speed
            x = throw_pos.x + horizontal_vel.x * t_fall
            y = throw_pos.y + horizontal_vel.y * t_fall
            z = throw_pos.z - 0.5 * self.g * t_fall * t_fall
            return Vector3GPU(x, y, z)
        else:
            # 烟球下沉阶段
            t_sink = t - boom_time
            horizontal_vel = drone_dir * drone_speed
            boom_x = throw_pos.x + horizontal_vel.x * fall_time
            boom_y = throw_pos.y + horizontal_vel.y * fall_time
            boom_z = throw_pos.z - 0.5 * self.g * fall_time * fall_time
            
            return Vector3GPU(boom_x, boom_y, boom_z - sink_speed * t_sink)
    
    def is_cylinder_in_shadow_cone(self, missile_pos, smoke_pos):
        """超高精度判断圆柱体是否被遮蔽"""
        MS = smoke_pos - missile_pos
        dist_MS = MS.length()
        
        if dist_MS <= self.SMOKE_RADIUS:
            return False
        
        # 计算圆锥参数
        sin_alpha = self.SMOKE_RADIUS / dist_MS
        cos_alpha = math.sqrt(1 - sin_alpha * sin_alpha)
        tan_alpha = sin_alpha / cos_alpha
        
        cone_axis = MS.normalize()
        
        # 检查采样点
        points_in_shadow = 0
        total_points = len(self.sampling_points)
        
        for i in range(total_points):
            if GPU_AVAILABLE:
                point = Vector3GPU(float(self.sampling_points[i][0]), 
                                 float(self.sampling_points[i][1]), 
                                 float(self.sampling_points[i][2]))
            else:
                point = Vector3GPU(self.sampling_points[i][0], 
                                 self.sampling_points[i][1], 
                                 self.sampling_points[i][2])
            
            MP = point - missile_pos
            proj_length = MP.dot(cone_axis)
            
            if proj_length > dist_MS:
                projection = cone_axis * proj_length
                perpendicular = MP - projection
                perp_distance = perpendicular.length()
                
                distance_from_smoke = proj_length - dist_MS
                cone_radius_at_point = distance_from_smoke * tan_alpha
                
                if perp_distance <= cone_radius_at_point + 0.05:
                    points_in_shadow += 1
        
        shadow_ratio = points_in_shadow / total_points
        return shadow_ratio > 0.12
    
    def calculate_blocking_time(self, params: Dict[str, float]) -> Tuple[float, Dict]:
        """计算遮蔽时间并返回轨迹数据"""
        drone_speed = params['droneSpeed']
        drone_direction = params['droneDirection']
        throw_time = params['throwTime']
        fall_time = params['fallTime']
        
        # 使用固定参数
        drone_init_x = self.DRONE_INIT_X
        drone_init_y = self.DRONE_INIT_Y
        sink_speed = self.SINK_SPEED
        
        # 参数验证
        if not (70 <= drone_speed <= 140):
            return 0, {}
        if not (0.5 <= throw_time <= 3.0):
            return 0, {}
        if not (2.0 <= fall_time <= 5.0):
            return 0, {}
        if not (1 <= sink_speed <= 5):
            return 0, {}
        
        # 无人机初始位置和方向
        D0 = Vector3GPU(drone_init_x, drone_init_y, self.DRONE_HEIGHT)
        direction_rad = math.radians(drone_direction)
        drone_dir = Vector3GPU(math.cos(direction_rad), math.sin(direction_rad), 0).normalize()
        
        boom_time = throw_time + fall_time
        block_start = None
        block_end = None
        was_blocked = False
        
        # 轨迹数据记录
        trajectory_data = {
            'missile_path': [],
            'drone_path': [],
            'smoke_path': [],
            'blocking_periods': [],
            'params': params
        }
        
        # 时间步长仿真
        max_time = boom_time + 15
        time_step = 0.01
        
        t = 0
        while t <= max_time:
            # 计算位置
            missile_pos = self.get_missile_position(t)
            
            # 无人机位置（只在投放前移动）
            if t <= throw_time:
                drone_pos = D0 + drone_dir * (drone_speed * t)
            else:
                drone_pos = D0 + drone_dir * (drone_speed * throw_time)
            
            # 记录轨迹
            trajectory_data['missile_path'].append({
                'time': t,
                'position': missile_pos.to_dict()
            })
            
            trajectory_data['drone_path'].append({
                'time': t,
                'position': drone_pos.to_dict()
            })
            
            # 烟雾位置和遮蔽检查
            if t >= boom_time:
                smoke_pos = self.get_smoke_position(t, D0, drone_dir, drone_speed, throw_time, fall_time, sink_speed)
                
                if smoke_pos and smoke_pos.z > -100:
                    trajectory_data['smoke_path'].append({
                        'time': t,
                        'position': smoke_pos.to_dict()
                    })
                    
                    # 遮蔽检查
                    is_blocked = self.is_cylinder_in_shadow_cone(missile_pos, smoke_pos)
                    
                    if is_blocked and not was_blocked:
                        block_start = t
                        was_blocked = True
                    elif not is_blocked and was_blocked:
                        block_end = t
                        trajectory_data['blocking_periods'].append({
                            'start': block_start,
                            'end': block_end,
                            'duration': block_end - block_start
                        })
                        break
                else:
                    break
            
            t += time_step
        
        # 计算总遮蔽时间
        total_block_time = 0
        if block_start is not None:
            if block_end is not None:
                total_block_time = block_end - block_start
            else:
                total_block_time = max_time - block_start
                trajectory_data['blocking_periods'].append({
                    'start': block_start,
                    'end': max_time,
                    'duration': total_block_time
                })
        
        return total_block_time, trajectory_data
    
    def create_random_individual(self) -> Dict[str, float]:
        """创建随机个体"""
        individual = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            if param == 'droneDirection':
                # 角度参数，使用整数
                individual[param] = random.randint(int(min_val), int(max_val))
            else:
                # 其他参数，使用浮点数
                individual[param] = random.uniform(min_val, max_val)
        return individual
    
    def initialize_population(self) -> List[Dict[str, float]]:
        """初始化种群"""
        return [self.create_random_individual() for _ in range(self.population_size)]
    
    def evaluate_population(self, population: List[Dict[str, float]]) -> List[float]:
        """评估种群适应度"""
        fitness_scores = []
        for individual in population:
            block_time, _ = self.calculate_blocking_time(individual)
            fitness_scores.append(block_time)
        return fitness_scores
    
    def tournament_selection(self, population: List[Dict[str, float]], 
                           fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, float]:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index].copy()
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        for param in self.param_ranges.keys():
            if random.random() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """变异操作"""
        mutated = individual.copy()
        for param, (min_val, max_val) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if param == 'droneDirection':
                    # 角度参数特殊处理
                    mutated[param] = random.randint(int(min_val), int(max_val))
                else:
                    # 高斯变异
                    current_val = mutated[param]
                    mutation_strength = (max_val - min_val) * 0.1
                    new_val = current_val + random.gauss(0, mutation_strength)
                    mutated[param] = max(min_val, min(max_val, new_val))
        return mutated
    
    def optimize(self, callback=None) -> OptimizationResult:
        """执行优化算法"""
        print("开始参数优化...")
        print(f"GPU状态: {'启用' if GPU_AVAILABLE else '未启用'}")
        print(f"种群大小: {self.population_size}")
        print(f"最大代数: {self.max_generations}")
        print(f"采样点数: {len(self.sampling_points)}")
        
        # 初始化
        population = self.initialize_population()
        best_fitness = 0
        best_individual = None
        best_trajectory = None
        fitness_history = []
        generation_without_improvement = 0
        
        for generation in range(self.max_generations):
            self.current_generation = generation
            
            # 评估种群
            fitness_scores = self.evaluate_population(population)
            
            # 找到当前最佳
            current_best_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            current_best_fitness = fitness_scores[current_best_index]
            current_best_individual = population[current_best_index]
            
            # 更新全局最佳
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()
                _, best_trajectory = self.calculate_blocking_time(best_individual)
                generation_without_improvement = 0
                
                print(f"第 {generation+1} 代: 新的最佳遮蔽时间 = {best_fitness:.4f}s")
                print(f"  最佳参数:")
                print(f"    无人机速度: {best_individual['droneSpeed']:.2f} m/s")
                print(f"    飞行方向: {best_individual['droneDirection']:.2f}°")
                print(f"    投放时间: {best_individual['throwTime']:.2f} s")
                print(f"    平抛时间: {best_individual['fallTime']:.2f} s")
                
                # 调用回调函数（用于实时更新）
                if callback:
                    callback(generation, best_fitness, best_individual, best_trajectory)
            else:
                generation_without_improvement += 1
            
            fitness_history.append(best_fitness)
            
            # 早停条件
            if generation_without_improvement > 50:
                print(f"\n连续50代无改进，提前停止优化")
                break
            
            # 进度输出
            if (generation + 1) % 25 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                print(f"第 {generation+1} 代: 平均遮蔽时间 = {avg_fitness:.4f}s, 最佳 = {best_fitness:.4f}s")
            
            # 生成下一代
            new_population = []
            
            # 精英保留
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            for i in elite_indices:
                new_population.append(population[i].copy())
            
            # 生成剩余个体
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # 创建结果对象
        result = OptimizationResult(
            block_time=best_fitness,
            params=best_individual,
            trajectory_data=best_trajectory,
            generation=generation,
            fitness_history=fitness_history
        )
        
        self.best_result = result
        return result
    
    def export_results_to_json(self, filename: str = "optimization_results.json"):
        """导出结果到JSON文件"""
        if self.best_result is None:
            print("没有优化结果可导出")
            return
        
        export_data = {
            'best_block_time': self.best_result.block_time,
            'best_params': self.best_result.params,
            'trajectory_data': self.best_result.trajectory_data,
            'optimization_info': {
                'generations': self.best_result.generation,
                'population_size': self.population_size,
                'gpu_enabled': GPU_AVAILABLE,
                'sampling_points': len(self.sampling_points)
            },
            'fitness_history': self.best_result.fitness_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"结果已导出到: {filename}")
    
    def print_results(self):
        """打印优化结果"""
        if self.best_result is None:
            print("没有优化结果")
            return
        
        print("\n" + "="*60)
        print("优化结果总结")
        print("="*60)
        print(f"最佳遮蔽时间: {self.best_result.block_time:.4f} 秒")
        print(f"优化代数: {self.best_result.generation}")
        print(f"GPU加速: {'启用' if GPU_AVAILABLE else '未启用'}")
        print(f"采样点数: {len(self.sampling_points)}")
        
        print("\n最佳参数:")
        for param, value in self.best_result.params.items():
            print(f"  {param}: {value:.4f}")
        
        if self.best_result.trajectory_data and 'blocking_periods' in self.best_result.trajectory_data:
            print("\n遮蔽时段:")
            for i, period in enumerate(self.best_result.trajectory_data['blocking_periods']):
                print(f"  时段 {i+1}: {period['start']:.3f}s - {period['end']:.3f}s (持续 {period['duration']:.3f}s)")
        
        print("="*60)

def main():
    """主函数"""
    print("导弹-烟球-圆柱遮蔽参数优化算法")
    print("目标: 最大化遮蔽时间")
    print("="*50)
    
    # 创建优化器
    optimizer = OptimizationAlgorithmInteractive()
    
    # 执行优化
    start_time = time.time()
    result = optimizer.optimize()
    end_time = time.time()
    
    # 输出结果
    optimizer.print_results()
    print(f"\n优化耗时: {end_time - start_time:.2f} 秒")
    
    # 导出结果
    optimizer.export_results_to_json("optimization_results.json")
    
    return result

if __name__ == "__main__":
    main()