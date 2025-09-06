#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速依赖安装脚本
自动检测CUDA版本并安装相应的CuPy包
"""

import subprocess
import sys
import os
import platform

def run_command(command):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def detect_cuda_version():
    """检测CUDA版本"""
    print("检测CUDA版本...")
    
    # 尝试通过nvidia-smi检测
    success, stdout, stderr = run_command("nvidia-smi")
    if success and "CUDA Version" in stdout:
        try:
            # 从nvidia-smi输出中提取CUDA版本
            lines = stdout.split('\n')
            for line in lines:
                if "CUDA Version" in line:
                    version_str = line.split("CUDA Version: ")[1].split()[0]
                    major_version = int(version_str.split('.')[0])
                    print(f"检测到CUDA版本: {version_str}")
                    return major_version
        except:
            pass
    
    # 尝试通过nvcc检测
    success, stdout, stderr = run_command("nvcc --version")
    if success and "release" in stdout:
        try:
            lines = stdout.split('\n')
            for line in lines:
                if "release" in line:
                    version_str = line.split("release ")[1].split(',')[0]
                    major_version = int(version_str.split('.')[0])
                    print(f"检测到CUDA版本: {version_str}")
                    return major_version
        except:
            pass
    
    print("未检测到CUDA，将安装CPU版本")
    return None

def install_cupy(cuda_version):
    """根据CUDA版本安装CuPy"""
    if cuda_version is None:
        print("跳过CuPy安装（未检测到CUDA）")
        return False
    
    # 根据CUDA版本选择CuPy包
    if cuda_version >= 12:
        package = "cupy-cuda12x"
    elif cuda_version >= 11:
        package = "cupy-cuda11x"
    elif cuda_version >= 10:
        package = "cupy-cuda110"  # CUDA 10.x使用cuda110包
    else:
        print(f"不支持的CUDA版本: {cuda_version}")
        return False
    
    print(f"安装{package}...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install {package}")
    
    if success:
        print(f"✓ {package} 安装成功")
        return True
    else:
        print(f"✗ {package} 安装失败")
        print(f"错误信息: {stderr}")
        return False

def install_other_deps():
    """安装其他依赖包"""
    packages = [
        "numpy",
        "typing-extensions"
    ]
    
    print("安装其他依赖包...")
    for package in packages:
        print(f"安装 {package}...")
        success, stdout, stderr = run_command(f"{sys.executable} -m pip install {package}")
        
        if success:
            print(f"✓ {package} 安装成功")
        else:
            print(f"✗ {package} 安装失败: {stderr}")

def test_gpu_acceleration():
    """测试GPU加速是否可用"""
    print("\n测试GPU加速...")
    
    test_code = """
try:
    import cupy as cp
    import numpy as np
    
    # 创建测试数组
    cpu_array = np.random.random((1000, 1000))
    gpu_array = cp.asarray(cpu_array)
    
    # 执行矩阵乘法测试
    result = cp.dot(gpu_array, gpu_array)
    
    # 获取GPU信息
    device = cp.cuda.Device()
    print(f"GPU设备: {device}")
    print(f"GPU内存: {device.mem_info[1] / 1024**3:.1f} GB")
    print("✓ GPU加速测试成功")
    
except ImportError:
    print("✗ CuPy未安装")
except Exception as e:
    print(f"✗ GPU加速测试失败: {e}")
"""
    
    success, stdout, stderr = run_command(f'{sys.executable} -c "{test_code}"')
    
    if success:
        print(stdout)
    else:
        print("GPU加速不可用，将使用CPU计算")
        print(stderr)

def main():
    """主函数"""
    print("=" * 60)
    print("GPU加速依赖安装脚本")
    print("=" * 60)
    
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print()
    
    # 检测CUDA版本
    cuda_version = detect_cuda_version()
    
    # 安装其他依赖
    install_other_deps()
    
    # 安装CuPy
    cupy_installed = install_cupy(cuda_version)
    
    # 测试GPU加速
    if cupy_installed:
        test_gpu_acceleration()
    
    print("\n=" * 60)
    print("安装完成")
    print("=" * 60)
    
    if cupy_installed:
        print("✓ GPU加速已启用")
        print("现在可以运行: python optimization_gpu.py")
    else:
        print("⚠ GPU加速未启用，将使用CPU计算")
        print("如需GPU加速，请确保:")
        print("1. 安装了NVIDIA GPU驱动")
        print("2. 安装了CUDA Toolkit")
        print("3. 重新运行此脚本")
    
    print("\n手动安装命令:")
    if cuda_version:
        if cuda_version >= 12:
            print("  pip install cupy-cuda12x")
        elif cuda_version >= 11:
            print("  pip install cupy-cuda11x")
        else:
            print("  pip install cupy-cuda110")
    else:
        print("  pip install cupy-cuda12x  # 最新CUDA版本")
        print("  pip install cupy-cuda11x  # CUDA 11.x")

if __name__ == "__main__":
    main()