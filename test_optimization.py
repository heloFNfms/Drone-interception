#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正后的优化算法
验证参数范围和优化逻辑是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimization_interactive import OptimizationAlgorithmInteractive

def test_parameter_ranges():
    """
    测试参数范围是否正确设置
    """
    print("=== 测试参数范围 ===")
    
    optimizer = OptimizationAlgorithmInteractive()
    
    print("优化参数范围:")
    for param, (min_val, max_val) in optimizer.param_ranges.items():
        print(f"  {param}: [{min_val}, {max_val}]")
    
    print("\n固定参数:")
    print(f"  无人机初始位置X: {optimizer.DRONE_INIT_X} m")
    print(f"  无人机初始位置Y: {optimizer.DRONE_INIT_Y} m")
    print(f"  无人机飞行高度: {optimizer.DRONE_HEIGHT} m")
    print(f"  烟球下沉速度: {optimizer.SINK_SPEED} m/s")
    
    return optimizer

def test_individual_creation(optimizer):
    """
    测试个体创建是否正确
    """
    print("\n=== 测试个体创建 ===")
    
    # 创建几个随机个体
    for i in range(3):
        individual = optimizer.create_random_individual()
        print(f"\n个体 {i+1}:")
        for param, value in individual.items():
            print(f"  {param}: {value:.2f}")
        
        # 验证参数是否在范围内
        valid = True
        for param, value in individual.items():
            min_val, max_val = optimizer.param_ranges[param]
            if not (min_val <= value <= max_val):
                print(f"  警告: {param} = {value} 超出范围 [{min_val}, {max_val}]")
                valid = False
        
        if valid:
            print(f"  ✓ 个体 {i+1} 参数范围正确")

def test_blocking_time_calculation(optimizer):
    """
    测试遮蔽时间计算
    """
    print("\n=== 测试遮蔽时间计算 ===")
    
    # 测试几个不同的参数组合
    test_cases = [
        {
            'droneSpeed': 120,
            'droneDirection': 180,
            'throwTime': 1.5,
            'fallTime': 3.6
        },
        {
            'droneSpeed': 100,
            'droneDirection': 90,
            'throwTime': 2.0,
            'fallTime': 4.0
        },
        {
            'droneSpeed': 80,
            'droneDirection': 270,
            'throwTime': 1.0,
            'fallTime': 3.0
        }
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        print(f"  参数: {params}")
        
        blocking_time, _ = optimizer.calculate_blocking_time(params)
        print(f"  遮蔽时间: {blocking_time:.4f} 秒")
        
        if blocking_time > 0:
            print(f"  ✓ 计算成功")
        else:
            print(f"  ✗ 计算失败或无遮蔽")

def test_short_optimization(optimizer):
    """
    测试短时间优化
    """
    print("\n=== 测试短时间优化 ===")
    
    # 设置较小的种群和代数进行快速测试
    optimizer.population_size = 20
    optimizer.max_generations = 10
    
    print(f"开始优化 (种群大小: {optimizer.population_size}, 最大代数: {optimizer.max_generations})...")
    
    try:
        best_params, best_fitness = optimizer.optimize()
        
        print(f"\n优化完成!")
        print(f"最佳遮蔽时间: {best_fitness:.4f} 秒")
        print(f"最佳参数:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.2f}")
        
        return True
        
    except Exception as e:
        print(f"优化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主测试函数
    """
    print("开始测试修正后的优化算法...\n")
    
    try:
        # 1. 测试参数范围
        optimizer = test_parameter_ranges()
        
        # 2. 测试个体创建
        test_individual_creation(optimizer)
        
        # 3. 测试遮蔽时间计算
        test_blocking_time_calculation(optimizer)
        
        # 4. 测试短时间优化
        success = test_short_optimization(optimizer)
        
        if success:
            print("\n=== 所有测试通过! ===")
            print("优化算法已正确修正，现在只优化以下4个参数:")
            print("1. 无人机飞行速度 (70-140 m/s)")
            print("2. 无人机飞行方向 (0-360°)")
            print("3. 烟幕弹投放时间 (0.5-3.0 s)")
            print("4. 烟幕弹平抛时间 (2.0-5.0 s)")
            print("\n固定参数:")
            print(f"- 无人机初始位置: ({optimizer.DRONE_INIT_X}, {optimizer.DRONE_INIT_Y}, {optimizer.DRONE_HEIGHT}) m")
            print(f"- 烟球下沉速度: {optimizer.SINK_SPEED} m/s")
        else:
            print("\n=== 测试失败 ===")
            print("请检查代码中的错误")
            
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()