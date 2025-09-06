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
    
    # 优先检查环境变量（云端环境常用）
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"检测到CUDA_HOME: {cuda_home}")
    
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
                    minor_version = int(version_str.split('.')[1])
                    print(f"检测到CUDA版本: {version_str}")
                    return major_version, minor_version
        except Exception as e:
            print(f"解析nvidia-smi输出失败: {e}")
    
    # 尝试通过nvcc检测
    success, stdout, stderr = run_command("nvcc --version")
    if success and "release" in stdout:
        try:
            lines = stdout.split('\n')
            for line in lines:
                if "release" in line:
                    version_str = line.split("release ")[1].split(',')[0]
                    major_version = int(version_str.split('.')[0])
                    minor_version = int(version_str.split('.')[1])
                    print(f"检测到CUDA版本: {version_str}")
                    return major_version, minor_version
        except Exception as e:
            print(f"解析nvcc输出失败: {e}")
    
    # 云端环境特殊检测
    if os.path.exists('/usr/local/cuda'):
        print("检测到云端CUDA环境，假设为CUDA 12.1")
        return 12, 1
    
    print("未检测到CUDA，将安装CPU版本")
    return None, None

def install_cupy(cuda_version_info):
    """根据CUDA版本安装CuPy"""
    if cuda_version_info[0] is None:
        print("跳过CuPy安装（未检测到CUDA）")
        return False
    
    major_version, minor_version = cuda_version_info
    
    # 根据CUDA版本选择CuPy包
    if major_version >= 12:
        package = "cupy-cuda12x"
    elif major_version >= 11:
        package = "cupy-cuda11x"
    elif major_version >= 10:
        package = "cupy-cuda110"  # CUDA 10.x使用cuda110包
    else:
        print(f"不支持的CUDA版本: {major_version}.{minor_version}")
        return False
    
    print(f"安装{package}...")
    
    # 云端环境优化：使用国内镜像源加速安装
    pip_commands = [
        f"{sys.executable} -m pip install {package} -i https://pypi.tuna.tsinghua.edu.cn/simple/",
        f"{sys.executable} -m pip install {package} -i https://mirrors.aliyun.com/pypi/simple/",
        f"{sys.executable} -m pip install {package}"
    ]
    
    for cmd in pip_commands:
        print(f"尝试: {cmd.split()[-1]}")
        success, stdout, stderr = run_command(cmd)
        
        if success:
            print(f"✓ {package} 安装成功")
            return True
        else:
            print(f"安装失败，尝试下一个源...")
    
    print(f"✗ {package} 安装失败")
    print(f"最后错误信息: {stderr}")
    return False

def install_other_deps():
    """安装其他依赖包"""
    packages = [
        "numpy",
        "typing-extensions",
        "matplotlib",  # 可能需要的可视化库
        "scipy"       # 科学计算库
    ]
    
    print("安装其他依赖包...")
    
    # 云端环境优化：批量安装以提高效率
    package_list = " ".join(packages)
    
    # 尝试使用国内镜像源
    pip_commands = [
        f"{sys.executable} -m pip install {package_list} -i https://pypi.tuna.tsinghua.edu.cn/simple/",
        f"{sys.executable} -m pip install {package_list} -i https://mirrors.aliyun.com/pypi/simple/",
        f"{sys.executable} -m pip install {package_list}"
    ]
    
    for cmd in pip_commands:
        print(f"尝试批量安装: {cmd.split('install')[1].split('-i')[0].strip()}")
        success, stdout, stderr = run_command(cmd)
        
        if success:
            print("✓ 所有依赖包安装成功")
            return
        else:
            print("批量安装失败，尝试单独安装...")
            break
    
    # 如果批量安装失败，逐个安装
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
    import time
    
    print("CuPy版本:", cp.__version__)
    
    # 获取GPU信息
    device = cp.cuda.Device()
    print(f"GPU设备ID: {device.id}")
    
    # 获取GPU属性
    with device:
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"GPU名称: {props['name'].decode()}")
        print(f"计算能力: {props['major']}.{props['minor']}")
        print(f"多处理器数量: {props['multiProcessorCount']}")
        
        # 获取内存信息
        mem_info = cp.cuda.runtime.memGetInfo()
        free_mem = mem_info[0] / 1024**3
        total_mem = mem_info[1] / 1024**3
        used_mem = total_mem - free_mem
        print(f"GPU内存: {used_mem:.1f}GB / {total_mem:.1f}GB (已用/总计)")
    
    # 性能测试
    print("\n执行性能测试...")
    size = 2000
    
    # CPU测试
    cpu_array = np.random.random((size, size)).astype(np.float32)
    start_time = time.time()
    cpu_result = np.dot(cpu_array, cpu_array)
    cpu_time = time.time() - start_time
    
    # GPU测试
    gpu_array = cp.asarray(cpu_array)
    cp.cuda.Stream.null.synchronize()  # 确保数据传输完成
    start_time = time.time()
    gpu_result = cp.dot(gpu_array, gpu_array)
    cp.cuda.Stream.null.synchronize()  # 确保计算完成
    gpu_time = time.time() - start_time
    
    print(f"CPU计算时间: {cpu_time:.3f}秒")
    print(f"GPU计算时间: {gpu_time:.3f}秒")
    print(f"加速比: {cpu_time/gpu_time:.1f}x")
    
    # 验证结果正确性
    cpu_result_gpu = cp.asnumpy(gpu_result)
    max_diff = np.max(np.abs(cpu_result - cpu_result_gpu))
    print(f"计算误差: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✓ GPU加速测试成功，结果正确")
    else:
        print("⚠ GPU计算结果与CPU有较大差异")
    
except ImportError as e:
    print(f"✗ CuPy导入失败: {e}")
except Exception as e:
    print(f"✗ GPU加速测试失败: {e}")
    import traceback
    traceback.print_exc()
"""
    
    success, stdout, stderr = run_command(f'{sys.executable} -c "{test_code}"')
    
    if success:
        print(stdout)
    else:
        print("GPU加速测试失败")
        print("标准输出:", stdout)
        print("错误输出:", stderr)

def main():
    """主函数"""
    print("=" * 60)
    print("GPU加速依赖安装脚本 (云端优化版)")
    print("=" * 60)
    
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    
    # 检查是否在云端环境
    cloud_indicators = [
        os.path.exists('/content'),  # Google Colab
        os.path.exists('/kaggle'),   # Kaggle
        'COLAB_GPU' in os.environ,   # Colab GPU
        'KAGGLE_KERNEL_RUN_TYPE' in os.environ  # Kaggle
    ]
    
    if any(cloud_indicators):
        print("检测到云端环境")
    
    print()
    
    # 检测CUDA版本
    cuda_version_info = detect_cuda_version()
    
    # 显示GPU信息
    print("\n检测GPU信息...")
    success, stdout, stderr = run_command("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
    if success:
        print(f"GPU信息: {stdout.strip()}")
    
    # 安装其他依赖
    install_other_deps()
    
    # 安装CuPy
    cupy_installed = install_cupy(cuda_version_info)
    
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
    if cuda_version_info[0]:
        major_version = cuda_version_info[0]
        if major_version >= 12:
            print("  pip install cupy-cuda12x -i https://pypi.tuna.tsinghua.edu.cn/simple/")
        elif major_version >= 11:
            print("  pip install cupy-cuda11x -i https://pypi.tuna.tsinghua.edu.cn/simple/")
        else:
            print("  pip install cupy-cuda110 -i https://pypi.tuna.tsinghua.edu.cn/simple/")
    else:
        print("  pip install cupy-cuda12x -i https://pypi.tuna.tsinghua.edu.cn/simple/  # 最新CUDA版本")
        print("  pip install cupy-cuda11x -i https://pypi.tuna.tsinghua.edu.cn/simple/  # CUDA 11.x")
    
    print("\n云端环境使用建议:")
    print("1. 确保选择了GPU实例")
    print("2. 检查CUDA驱动是否正确安装")
    print("3. 如果安装失败，尝试重启内核后重新运行")
    print("4. 可以使用 'nvidia-smi' 命令检查GPU状态")

if __name__ == "__main__":
    main()