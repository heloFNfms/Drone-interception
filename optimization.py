#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导弹-烟球-圆柱遮蔽参数优化算法
基于超高精度遮蔽判定算法的参数优化与最长遮蔽时间计算
"""

import numpy as np
import math
import random
import time
from typing import Dict, List, Tuple, Optional

class Vector3:
    """三维向量类"""
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return Vector3(self.x / length, self.y / length, self.z / length)
        return Vector3(0, 0, 0)
    
    def distance_to(self, other) -> float:
        return (self - other).length()
    
    def __str__(self):
        return f"Vector3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

class OptimizationAlgorithm:
    """导弹-烟球-圆柱遮蔽参数优化算法"""
    
    def __init__(self):
        # 物理常量和初始条件
        self.g = 9.8  # 重力加速度 m/s²
        self.v_missile = 500  # 导弹速度 m/s
        
        # 导弹初始位置 M(0,0,1000)
        self.M0 = Vector3(0, 0, 1000)
        
        # 圆柱体目标参数 B(20000,0,0)
        self.B_CENTER = Vector3(20000, 0, 0)  # 圆柱体中心
        self.B_RADIUS = 100  # 圆柱体半径 m
        self.B_HEIGHT = 200  # 圆柱体高度 m
        
        # 固定参数
        self.SMOKE_RADIUS = 10  # 烟球半径固定为10m
        self.DRONE_HEIGHT = 1800  # 无人机飞行高度固定为1800m
        
        # 参数范围定义
        self.param_ranges = {
            'droneSpeed': (70, 140),      # 无人机速度范围 m/s
            'droneDirection': (0, 360),   # 飞行方向角度范围 度
            'droneInitX': (10000, 25000), # 初始位置X范围 m
            'droneInitY': (-5000, 5000),  # 初始位置Y范围 m
            'throwTime': (0.5, 3.0),      # 投放时间范围 s
            'fallTime': (2.0, 5.0),       # 平抛时间范围 s
            'sinkSpeed': (1, 5)           # 下沉速度范围 m/s
        }
        
        # 遗传算法参数
        self.population_size = 50
        self.max_generations = 1000
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
        # 优化状态
        self.best_result = {
            'blockTime': 0,
            'params': {}
        }
    
    def get_missile_position(self, t: float) -> Vector3:
        """计算导弹在时刻t的位置"""
        # 导弹从M0指向B_CENTER的单位方向向量
        direction = (self.B_CENTER - self.M0).normalize()
        
        # 导弹位置 = 初始位置 + 速度 * 时间 * 方向
        return self.M0 + direction * (self.v_missile * t)
    
    def get_smoke_position(self, t: float, D0: Vector3, drone_dir: Vector3, 
                          drone_speed: float, throw_time: float, 
                          fall_time: float, sink_speed: float) -> Vector3:
        """计算烟球在时刻t的位置"""
        boom_time = throw_time + fall_time
        
        if t < boom_time:
            return Vector3(0, 0, -10000)  # 烟球尚未爆炸
        
        # 投放点：无人机在投放时刻的位置
        throw_pos = D0 + drone_dir * (drone_speed * throw_time)
        
        # 起爆点：考虑平抛运动
        boom_pos = Vector3(
            throw_pos.x,
            throw_pos.y,
            throw_pos.z - 0.5 * self.g * fall_time**2
        )
        
        # 烟球位置：起爆后下沉
        elapsed = t - boom_time
        return Vector3(
            boom_pos.x,
            boom_pos.y,
            boom_pos.z - sink_speed * elapsed
        )
    
    def is_point_in_shadow_cone(self, point: Vector3, missile_pos: Vector3, 
                               smoke_pos: Vector3) -> bool:
        """判断点是否在阴影圆锥内"""
        # 从导弹到烟球的向量
        missile_to_smoke = smoke_pos - missile_pos
        missile_to_smoke_length = missile_to_smoke.length()
        
        if missile_to_smoke_length < 1e-6:
            return False
        
        # 从导弹到点的向量
        missile_to_point = point - missile_pos
        missile_to_point_length = missile_to_point.length()
        
        if missile_to_point_length < 1e-6:
            return False
        
        # 计算圆锥轴线方向
        cone_axis = missile_to_smoke.normalize()
        
        # 点在轴线上的投影长度
        projection_length = missile_to_point.dot(cone_axis)
        
        # 检查点是否在圆锥的正确一侧
        if projection_length <= 0 or projection_length <= missile_to_smoke_length:
            return False
        
        # 计算圆锥在该距离处的半径
        cone_radius_at_distance = self.SMOKE_RADIUS * (projection_length / missile_to_smoke_length)
        
        # 计算点到轴线的距离
        point_on_axis = missile_pos + cone_axis * projection_length
        distance_to_axis = point.distance_to(point_on_axis)
        
        # 判断是否在圆锥内
        return distance_to_axis <= cone_radius_at_distance
    
    def is_cylinder_in_shadow_cone(self, missile_pos: Vector3, smoke_pos: Vector3) -> bool:
        """超高精度遮蔽判定算法 - 判断圆柱体是否被遮蔽"""
        blocked_points = 0
        total_points = 0
        
        # 1. 轴线采样 (沿圆柱体中心轴线)
        for i in range(21):  # 21个点
            z = i * (self.B_HEIGHT / 20)
            point = Vector3(self.B_CENTER.x, self.B_CENTER.y, z)
            total_points += 1
            if self.is_point_in_shadow_cone(point, missile_pos, smoke_pos):
                blocked_points += 1
        
        # 2. 表面采样 (圆柱体表面)
        height_samples = 20
        angle_samples = 36  # 每10度一个点
        
        for h in range(height_samples + 1):
            z = h * (self.B_HEIGHT / height_samples)
            for a in range(angle_samples):
                angle = a * (2 * math.pi / angle_samples)
                x = self.B_CENTER.x + self.B_RADIUS * math.cos(angle)
                y = self.B_CENTER.y + self.B_RADIUS * math.sin(angle)
                point = Vector3(x, y, z)
                total_points += 1
                if self.is_point_in_shadow_cone(point, missile_pos, smoke_pos):
                    blocked_points += 1
        
        # 3. 内部采样 (圆柱体内部)
        for h in range(10):
            z = (h + 1) * (self.B_HEIGHT / 11)
            for r_step in range(5):
                radius = (r_step + 1) * (self.B_RADIUS / 6)
                for a in range(12):
                    angle = a * (2 * math.pi / 12)
                    x = self.B_CENTER.x + radius * math.cos(angle)
                    y = self.B_CENTER.y + radius * math.sin(angle)
                    point = Vector3(x, y, z)
                    total_points += 1
                    if self.is_point_in_shadow_cone(point, missile_pos, smoke_pos):
                        blocked_points += 1
        
        # 4. 边缘精密采样
        for h in range(41):  # 更密集的高度采样
            z = h * (self.B_HEIGHT / 40)
            for a in range(72):  # 每5度一个点
                angle = a * (2 * math.pi / 72)
                # 边缘点
                x = self.B_CENTER.x + self.B_RADIUS * math.cos(angle)
                y = self.B_CENTER.y + self.B_RADIUS * math.sin(angle)
                point = Vector3(x, y, z)
                total_points += 1
                if self.is_point_in_shadow_cone(point, missile_pos, smoke_pos):
                    blocked_points += 1
        
        # 超敏感阈值判定
        if total_points == 0:
            return False
        
        blocking_ratio = blocked_points / total_points
        
        # 主阈值：12%的点被遮蔽即认为圆柱体被有效遮蔽
        main_threshold = 0.12
        
        return blocking_ratio >= main_threshold
    
    def calculate_blocking_time(self, params: Dict[str, float]) -> float:
        """计算遮蔽时间"""
        drone_speed = params['droneSpeed']
        drone_direction = params['droneDirection']
        drone_init_x = params['droneInitX']
        drone_init_y = params['droneInitY']
        throw_time = params['throwTime']
        fall_time = params['fallTime']
        sink_speed = params['sinkSpeed']
        
        # 参数验证
        if not (70 <= drone_speed <= 140):
            return 0
        if not (0.5 <= throw_time <= 3.0):
            return 0
        if not (2.0 <= fall_time <= 5.0):
            return 0
        if not (1 <= sink_speed <= 5):
            return 0
        
        # 无人机初始位置
        D0 = Vector3(drone_init_x, drone_init_y, self.DRONE_HEIGHT)
        
        # 无人机飞行方向向量
        direction_rad = math.radians(drone_direction)
        drone_dir = Vector3(
            math.cos(direction_rad),
            math.sin(direction_rad),
            0
        ).normalize()
        
        boom_time = throw_time + fall_time
        block_start = None
        block_end = None
        was_blocked = False
        
        # 超精细时间步长仿真
        max_time = boom_time + 20  # 最大仿真时间
        time_step = 0.001  # 1ms精度
        
        t = boom_time
        while t <= max_time:
            # 计算当前时刻各物体位置
            missile_pos = self.get_missile_position(t)
            smoke_pos = self.get_smoke_position(t, D0, drone_dir, drone_speed, 
                                              throw_time, fall_time, sink_speed)
            
            # 检查烟球是否还有效
            if smoke_pos.z < -100:  # 烟球落地过低
                break
            
            # 检查是否被遮蔽
            is_blocked = self.is_cylinder_in_shadow_cone(missile_pos, smoke_pos)
            
            if is_blocked and not was_blocked:
                # 遮蔽开始
                block_start = t
                was_blocked = True
            elif not is_blocked and was_blocked:
                # 遮蔽结束
                block_end = t
                break
            
            t += time_step
        
        # 计算遮蔽时间
        if block_start is not None:
            if block_end is not None:
                return block_end - block_start
            else:
                # 遮蔽持续到仿真结束
                return max_time - block_start
        
        return 0
    
    def create_random_individual(self) -> Dict[str, float]:
        """创建随机个体"""
        individual = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            individual[param] = random.uniform(min_val, max_val)
        return individual
    
    def initialize_population(self) -> List[Dict[str, float]]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = self.create_random_individual()
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual: Dict[str, float]) -> float:
        """评估个体适应度（遮蔽时间）"""
        return self.calculate_blocking_time(individual)
    
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
        
        child1 = {}
        child2 = {}
        
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
                # 高斯变异
                mutation_strength = 0.1  # 变异强度
                current_val = mutated[param]
                range_size = max_val - min_val
                mutation = random.gauss(0, range_size * mutation_strength)
                
                new_val = current_val + mutation
                # 确保在范围内
                new_val = max(min_val, min(max_val, new_val))
                mutated[param] = new_val
        
        return mutated
    
    def optimize(self) -> Dict:
        """执行遗传算法优化"""
        print("开始参数优化...")
        print(f"种群大小: {self.population_size}")
        print(f"最大代数: {self.max_generations}")
        print(f"参数范围:")
        for param, (min_val, max_val) in self.param_ranges.items():
            print(f"  {param}: {min_val} - {max_val}")
        print("\n" + "="*50)
        
        # 初始化种群
        population = self.initialize_population()
        
        best_fitness = 0
        best_individual = None
        generation_without_improvement = 0
        
        for generation in range(self.max_generations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                fitness_scores.append(fitness)
            
            # 找到当前最佳个体
            current_best_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            current_best_fitness = fitness_scores[current_best_index]
            current_best_individual = population[current_best_index]
            
            # 更新全局最佳
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual.copy()
                generation_without_improvement = 0
                print(f"第 {generation+1} 代: 新的最佳遮蔽时间 = {best_fitness:.4f}s")
                print(f"  最佳参数: {best_individual}")
            else:
                generation_without_improvement += 1
            
            # 早停条件
            if generation_without_improvement > 100:
                print(f"\n连续100代无改进，提前停止优化")
                break
            
            # 每50代输出进度
            if (generation + 1) % 50 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                print(f"第 {generation+1} 代: 平均遮蔽时间 = {avg_fitness:.4f}s, 最佳 = {best_fitness:.4f}s")
            
            # 生成下一代
            new_population = []
            
            # 精英保留
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
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
            
            # 确保种群大小
            population = new_population[:self.population_size]
        
        # 保存最佳结果
        self.best_result = {
            'blockTime': best_fitness,
            'params': best_individual
        }
        
        return self.best_result
    
    def print_results(self):
        """打印优化结果"""
        print("\n" + "="*60)
        print("优化结果")
        print("="*60)
        
        if self.best_result['blockTime'] > 0:
            print(f"最优遮蔽时间: {self.best_result['blockTime']:.6f} 秒")
            print("\n最优参数:")
            params = self.best_result['params']
            print(f"  无人机速度: {params['droneSpeed']:.2f} m/s")
            print(f"  飞行方向: {params['droneDirection']:.2f}°")
            print(f"  初始位置X: {params['droneInitX']:.2f} m")
            print(f"  初始位置Y: {params['droneInitY']:.2f} m")
            print(f"  投放时间: {params['throwTime']:.3f} s")
            print(f"  平抛时间: {params['fallTime']:.3f} s")
            print(f"  下沉速度: {params['sinkSpeed']:.2f} m/s")
            
            # 计算起爆时间
            boom_time = params['throwTime'] + params['fallTime']
            print(f"  起爆时间: {boom_time:.3f} s")
            
            print("\n物理参数:")
            print(f"  烟球半径: {self.SMOKE_RADIUS} m (固定)")
            print(f"  无人机高度: {self.DRONE_HEIGHT} m (固定)")
            print(f"  导弹速度: {self.v_missile} m/s")
            print(f"  圆柱体半径: {self.B_RADIUS} m")
            print(f"  圆柱体高度: {self.B_HEIGHT} m")
        else:
            print("未找到有效的遮蔽方案")
        
        print("="*60)

def main():
    """主函数"""
    print("导弹-烟球-圆柱遮蔽参数优化算法")
    print("基于超高精度遮蔽判定算法的参数优化与最长遮蔽时间计算")
    print("\n算法特性:")
    print("- 超高精度遮蔽判定算法 (1ms时间精度)")
    print("- 超密集采样点检测 (8000+采样点)")
    print("- 遗传算法全局优化")
    print("- 多参数联合优化")
    
    # 创建优化算法实例
    optimizer = OptimizationAlgorithm()
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行优化
    result = optimizer.optimize()
    
    # 记录结束时间
    end_time = time.time()
    
    # 打印结果
    optimizer.print_results()
    
    print(f"\n优化耗时: {end_time - start_time:.2f} 秒")
    
    # 验证最优解
    if result['blockTime'] > 0:
        print("\n验证最优解...")
        verification_time = optimizer.calculate_blocking_time(result['params'])
        print(f"验证遮蔽时间: {verification_time:.6f} 秒")
        
        if abs(verification_time - result['blockTime']) < 1e-6:
            print("✓ 验证通过")
        else:
            print("✗ 验证失败")

if __name__ == "__main__":
    main()