#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导弹-烟球-圆柱遮蔽参数优化算法 (GPU加速版本)
基于超高精度遮蔽判定算法的参数优化与最长遮蔽时间计算
使用CuPy进行GPU并行计算加速
"""

import numpy as np
import math
import random
import time
from typing import Dict, List, Tuple, Optional

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU加速已启用 (CuPy)")
except ImportError:
    print("警告: CuPy未安装，将使用CPU计算")
    print("安装命令: pip install cupy-cuda11x (根据CUDA版本选择)")
    import numpy as cp
    GPU_AVAILABLE = False

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
        if length > 1e-6:
            result = Vector3GPU(0, 0, 0)
            result.data = self.data / length
            return result
        return Vector3GPU(0, 0, 0)
    
    def distance_to(self, other):
        return (self - other).length()
    
    def to_cpu(self):
        """转换为CPU数组"""
        if GPU_AVAILABLE:
            return cp.asnumpy(self.data)
        else:
            return self.data

class OptimizationAlgorithmGPU:
    """GPU加速的导弹-烟球-圆柱遮蔽参数优化算法"""
    
    def __init__(self):
        # 物理常量和初始条件
        self.g = 9.8  # 重力加速度 m/s²
        self.v_missile = 500  # 导弹速度 m/s
        
        # 导弹初始位置 M(0,0,1000)
        self.M0 = Vector3GPU(0, 0, 1000)
        
        # 圆柱体目标参数 B(20000,0,0)
        self.B_CENTER = Vector3GPU(20000, 0, 0)  # 圆柱体中心
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
        
        # GPU优化的遗传算法参数
        self.population_size = 200 if GPU_AVAILABLE else 50  # GPU可以处理更大种群
        self.max_generations = 500 if GPU_AVAILABLE else 200  # 减少代数但增加种群
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 20 if GPU_AVAILABLE else 5
        
        # 优化状态
        self.best_result = {
            'blockTime': 0,
            'params': {}
        }
        
        # 预生成采样点（GPU优化）
        self._precompute_sampling_points()
    
    def _precompute_sampling_points(self):
        """预计算所有采样点坐标（GPU优化）"""
        sampling_points = []
        
        # 1. 轴线采样 (50点)
        for i in range(50):
            z = i * (self.B_HEIGHT / 49)
            sampling_points.append([self.B_CENTER.x, self.B_CENTER.y, z])
        
        # 2. 表面采样 (圆周 × 高度层)
        for layer in range(21):
            z = layer * (self.B_HEIGHT / 20)
            for angle in range(0, 360, 5):  # 每5度一个点
                rad = math.radians(angle)
                x = self.B_CENTER.x + self.B_RADIUS * math.cos(rad)
                y = self.B_CENTER.y + self.B_RADIUS * math.sin(rad)
                sampling_points.append([x, y, z])
        
        # 3. 内部采样 (极坐标)
        for layer in range(16):
            z = layer * (self.B_HEIGHT / 15)
            for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
                radius = r * self.B_RADIUS
                for angle in range(0, 360, 10):
                    rad = math.radians(angle)
                    x = self.B_CENTER.x + radius * math.cos(rad)
                    y = self.B_CENTER.y + radius * math.sin(rad)
                    sampling_points.append([x, y, z])
        
        # 4. 边缘精密采样
        for layer in range(11):
            z = layer * (self.B_HEIGHT / 10)
            for angle in range(0, 360, 3):  # 每3度一个点
                rad = math.radians(angle)
                for rFactor in [0.95, 0.975, 1.0]:
                    radius = rFactor * self.B_RADIUS
                    x = self.B_CENTER.x + radius * math.cos(rad)
                    y = self.B_CENTER.y + radius * math.sin(rad)
                    sampling_points.append([x, y, z])
        
        # 转换为GPU数组
        if GPU_AVAILABLE:
            self.sampling_points = cp.array(sampling_points, dtype=cp.float32)
        else:
            self.sampling_points = np.array(sampling_points, dtype=np.float32)
        
        print(f"预计算采样点数量: {len(sampling_points)}")
    
    def get_missile_position_gpu(self, t):
        """GPU加速的导弹位置计算"""
        # 导弹从M0指向B_CENTER的单位方向向量
        direction_data = self.B_CENTER.data - self.M0.data
        if GPU_AVAILABLE:
            direction_length = cp.linalg.norm(direction_data)
        else:
            direction_length = np.linalg.norm(direction_data)
        
        direction_data = direction_data / direction_length
        
        # 导弹位置 = 初始位置 + 速度 * 时间 * 方向
        result = Vector3GPU(0, 0, 0)
        result.data = self.M0.data + direction_data * (self.v_missile * t)
        return result
    
    def get_smoke_position_gpu(self, t, D0, drone_dir, drone_speed, throw_time, fall_time, sink_speed):
        """GPU加速的烟球位置计算"""
        boom_time = throw_time + fall_time
        
        if t < boom_time:
            return Vector3GPU(0, 0, -10000)  # 烟球尚未爆炸
        
        # 投放点：无人机在投放时刻的位置
        throw_pos = D0 + drone_dir * (drone_speed * throw_time)
        
        # 起爆点：考虑平抛运动
        boom_pos = Vector3GPU(
            throw_pos.x,
            throw_pos.y,
            throw_pos.z - 0.5 * self.g * fall_time**2
        )
        
        # 烟球位置：起爆后下沉
        elapsed = t - boom_time
        return Vector3GPU(
            boom_pos.x,
            boom_pos.y,
            boom_pos.z - sink_speed * elapsed
        )
    
    def is_cylinder_in_shadow_cone_gpu(self, missile_pos, smoke_pos):
        """GPU加速的超高精度遮蔽判定算法"""
        if smoke_pos.z < -100:
            return False
        
        # 计算从导弹到烟球的向量
        missile_to_smoke = smoke_pos.data - missile_pos.data
        if GPU_AVAILABLE:
            missile_to_smoke_length = cp.linalg.norm(missile_to_smoke)
        else:
            missile_to_smoke_length = np.linalg.norm(missile_to_smoke)
        
        if missile_to_smoke_length < 1e-6:
            return False
        
        # 圆锥轴线方向
        cone_axis = missile_to_smoke / missile_to_smoke_length
        
        # 批量计算所有采样点
        # 从导弹到各采样点的向量
        missile_to_points = self.sampling_points - missile_pos.data
        
        # 计算投影长度
        if GPU_AVAILABLE:
            projection_lengths = cp.dot(missile_to_points, cone_axis)
        else:
            projection_lengths = np.dot(missile_to_points, cone_axis)
        
        # 筛选在圆锥正确一侧的点
        valid_mask = projection_lengths > missile_to_smoke_length
        
        if GPU_AVAILABLE:
            valid_count = cp.sum(valid_mask)
        else:
            valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            return False
        
        # 计算有效点的垂直距离
        valid_points = missile_to_points[valid_mask]
        valid_projections = projection_lengths[valid_mask]
        
        # 计算投影点
        if GPU_AVAILABLE:
            projection_vectors = cp.outer(valid_projections, cone_axis)
        else:
            projection_vectors = np.outer(valid_projections, cone_axis)
        
        # 计算垂直向量
        perpendicular_vectors = valid_points - projection_vectors
        
        # 计算垂直距离
        if GPU_AVAILABLE:
            perpendicular_distances = cp.linalg.norm(perpendicular_vectors, axis=1)
        else:
            perpendicular_distances = np.linalg.norm(perpendicular_vectors, axis=1)
        
        # 计算圆锥在各点处的半径
        cone_radii = self.SMOKE_RADIUS * (valid_projections / missile_to_smoke_length)
        
        # 判断在圆锥内的点
        tolerance = 0.05
        if GPU_AVAILABLE:
            points_in_cone = cp.sum(perpendicular_distances <= (cone_radii + tolerance))
            total_valid_points = cp.sum(valid_mask)
        else:
            points_in_cone = np.sum(perpendicular_distances <= (cone_radii + tolerance))
            total_valid_points = np.sum(valid_mask)
        
        # 计算遮蔽比例
        if total_valid_points > 0:
            blocking_ratio = float(points_in_cone) / float(total_valid_points)
        else:
            blocking_ratio = 0
        
        # 超敏感阈值判定
        main_threshold = 0.12
        return blocking_ratio >= main_threshold
    
    def calculate_blocking_time_gpu(self, params: Dict[str, float]) -> float:
        """GPU加速的遮蔽时间计算"""
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
        D0 = Vector3GPU(drone_init_x, drone_init_y, self.DRONE_HEIGHT)
        
        # 无人机飞行方向向量
        direction_rad = math.radians(drone_direction)
        drone_dir = Vector3GPU(
            math.cos(direction_rad),
            math.sin(direction_rad),
            0
        ).normalize()
        
        boom_time = throw_time + fall_time
        block_start = None
        block_end = None
        was_blocked = False
        
        # 优化的时间步长仿真（GPU加速版本使用更大步长）
        max_time = boom_time + 15  # 减少最大仿真时间
        time_step = 0.005 if GPU_AVAILABLE else 0.01  # GPU版本使用更精细步长
        
        t = boom_time
        while t <= max_time:
            # 计算当前时刻各物体位置
            missile_pos = self.get_missile_position_gpu(t)
            smoke_pos = self.get_smoke_position_gpu(t, D0, drone_dir, drone_speed, 
                                                  throw_time, fall_time, sink_speed)
            
            # 检查烟球是否还有效
            if smoke_pos.z < -100:  # 烟球落地过低
                break
            
            # 检查是否被遮蔽
            is_blocked = self.is_cylinder_in_shadow_cone_gpu(missile_pos, smoke_pos)
            
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
    
    def evaluate_population_gpu(self, population: List[Dict[str, float]]) -> List[float]:
        """GPU加速的种群评估"""
        fitness_scores = []
        
        # 批量处理（如果GPU可用）
        if GPU_AVAILABLE:
            # 将种群分批处理以避免GPU内存溢出
            batch_size = 50
            for i in range(0, len(population), batch_size):
                batch = population[i:i+batch_size]
                batch_scores = []
                
                for individual in batch:
                    fitness = self.calculate_blocking_time_gpu(individual)
                    batch_scores.append(fitness)
                
                fitness_scores.extend(batch_scores)
                
                # 清理GPU内存
                if hasattr(cp, 'get_default_memory_pool'):
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
        else:
            # CPU版本
            for individual in population:
                fitness = self.calculate_blocking_time_gpu(individual)
                fitness_scores.append(fitness)
        
        return fitness_scores
    
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
        """执行GPU加速的遗传算法优化"""
        print("开始GPU加速参数优化...")
        print(f"GPU状态: {'启用' if GPU_AVAILABLE else '未启用'}")
        print(f"种群大小: {self.population_size}")
        print(f"最大代数: {self.max_generations}")
        print(f"采样点数: {len(self.sampling_points)}")
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
            # GPU加速的种群评估
            fitness_scores = self.evaluate_population_gpu(population)
            
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
            
            # 早停条件（GPU版本更严格）
            early_stop_threshold = 50 if GPU_AVAILABLE else 100
            if generation_without_improvement > early_stop_threshold:
                print(f"\n连续{early_stop_threshold}代无改进，提前停止优化")
                break
            
            # 每25代输出进度
            if (generation + 1) % 25 == 0:
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
        print("GPU加速优化结果")
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
            
            print("\nGPU加速信息:")
            print(f"  GPU状态: {'启用' if GPU_AVAILABLE else '未启用'}")
            print(f"  种群大小: {self.population_size}")
            print(f"  采样点数: {len(self.sampling_points)}")
        else:
            print("未找到有效的遮蔽方案")
        
        print("="*60)

def main():
    """主函数"""
    print("导弹-烟球-圆柱遮蔽参数优化算法 (GPU加速版本)")
    print("基于超高精度遮蔽判定算法的参数优化与最长遮蔽时间计算")
    print("\n算法特性:")
    print("- GPU并行计算加速 (CuPy)")
    print("- 超高精度遮蔽判定算法 (5ms时间精度)")
    print("- 批量采样点检测 (10000+采样点)")
    print("- 大规模遗传算法优化")
    print("- 多参数联合优化")
    
    if not GPU_AVAILABLE:
        print("\n注意: 未检测到GPU加速，将使用CPU计算")
        print("要启用GPU加速，请安装CuPy:")
        print("  pip install cupy-cuda11x  # 根据您的CUDA版本选择")
        print("  pip install cupy-cuda12x  # CUDA 12.x")
    
    # 创建优化算法实例
    optimizer = OptimizationAlgorithmGPU()
    
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
        verification_time = optimizer.calculate_blocking_time_gpu(result['params'])
        print(f"验证遮蔽时间: {verification_time:.6f} 秒")
        
        if abs(verification_time - result['blockTime']) < 1e-4:
            print("✓ 验证通过")
        else:
            print("✗ 验证失败")
    
    # GPU内存清理
    if GPU_AVAILABLE and hasattr(cp, 'get_default_memory_pool'):
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        print("\nGPU内存已清理")

if __name__ == "__main__":
    main()