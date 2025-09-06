#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端GPU环境快速配置脚本
专为云端GPU实例（如阿里云、腾讯云、AWS等）优化
"""

import subprocess
import sys
import os
import time

def run_command(command, timeout=300):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "命令执行超时"
    except Exception as e:
        return False, "", str(e)

def check_gpu_environment():
    """检查GPU环境"""
    print("🔍 检查GPU环境...")
    
    # 检查nvidia-smi
    success, stdout, stderr = run_command("nvidia-smi")
    if not success:
        print("❌ 未检测到NVIDIA GPU或驱动未安装")
        print("请确保您选择的是GPU实例")
        return False
    
    print("✅ GPU驱动检测成功")
    
    # 解析GPU信息
    lines = stdout.split('\n')
    for line in lines:
        if "CUDA Version" in line:
            cuda_version = line.split("CUDA Version: ")[1].split()[0]
            print(f"📋 CUDA版本: {cuda_version}")
            break
    
    # 获取GPU详细信息
    success, stdout, stderr = run_command("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
    if success:
        gpu_info = stdout.strip().split(', ')
        if len(gpu_info) >= 3:
            print(f"🎮 GPU型号: {gpu_info[0]}")
            print(f"💾 显存大小: {int(gpu_info[1])/1024:.1f} GB")
            print(f"🔧 驱动版本: {gpu_info[2]}")
    
    return True

def install_dependencies():
    """安装依赖包"""
    print("\n📦 安装依赖包...")
    
    # 更新pip
    print("更新pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # 基础依赖包
    base_packages = [
        "numpy",
        "scipy", 
        "matplotlib",
        "typing-extensions"
    ]
    
    # 使用清华镜像源批量安装
    package_list = " ".join(base_packages)
    print(f"安装基础包: {package_list}")
    
    success, stdout, stderr = run_command(
        f"{sys.executable} -m pip install {package_list} -i https://pypi.tuna.tsinghua.edu.cn/simple/ --timeout 120"
    )
    
    if success:
        print("✅ 基础依赖包安装成功")
    else:
        print("⚠️ 基础包安装失败，尝试逐个安装...")
        for package in base_packages:
            success, _, _ = run_command(f"{sys.executable} -m pip install {package}")
            if success:
                print(f"✅ {package} 安装成功")
            else:
                print(f"❌ {package} 安装失败")

def install_cupy():
    """安装CuPy GPU加速库"""
    print("\n🚀 安装CuPy GPU加速库...")
    
    # 检测CUDA版本
    success, stdout, stderr = run_command("nvidia-smi")
    if success and "CUDA Version" in stdout:
        for line in stdout.split('\n'):
            if "CUDA Version" in line:
                cuda_version = line.split("CUDA Version: ")[1].split()[0]
                major_version = int(cuda_version.split('.')[0])
                break
    else:
        print("⚠️ 无法检测CUDA版本，默认使用CUDA 12.x")
        major_version = 12
    
    # 选择合适的CuPy版本
    if major_version >= 12:
        cupy_package = "cupy-cuda12x"
    elif major_version >= 11:
        cupy_package = "cupy-cuda11x"
    else:
        cupy_package = "cupy-cuda110"
    
    print(f"安装 {cupy_package}...")
    
    # 尝试多个镜像源
    mirrors = [
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://pypi.douban.com/simple/"
    ]
    
    for mirror in mirrors:
        print(f"尝试镜像源: {mirror}")
        success, stdout, stderr = run_command(
            f"{sys.executable} -m pip install {cupy_package} -i {mirror} --timeout 300",
            timeout=400
        )
        
        if success:
            print(f"✅ {cupy_package} 安装成功")
            return True
        else:
            print(f"❌ 安装失败: {stderr[:200]}...")
    
    # 最后尝试官方源
    print("尝试官方PyPI源...")
    success, stdout, stderr = run_command(
        f"{sys.executable} -m pip install {cupy_package} --timeout 300",
        timeout=400
    )
    
    if success:
        print(f"✅ {cupy_package} 安装成功")
        return True
    else:
        print(f"❌ CuPy安装失败: {stderr}")
        return False

def test_gpu_performance():
    """测试GPU性能"""
    print("\n🧪 测试GPU性能...")
    
    test_script = '''
import time
import numpy as np

try:
    import cupy as cp
    
    print(f"CuPy版本: {cp.__version__}")
    
    # GPU信息
    device = cp.cuda.Device()
    with device:
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"GPU: {props["name"].decode()}")
        
        mem_info = cp.cuda.runtime.memGetInfo()
        total_mem = mem_info[1] / 1024**3
        free_mem = mem_info[0] / 1024**3
        print(f"显存: {total_mem:.1f}GB 总计, {free_mem:.1f}GB 可用")
    
    # 性能基准测试
    print("\n执行矩阵乘法基准测试...")
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        print(f"\n测试矩阵大小: {size}x{size}")
        
        # 生成测试数据
        np.random.seed(42)
        a = np.random.random((size, size)).astype(np.float32)
        b = np.random.random((size, size)).astype(np.float32)
        
        # CPU测试
        start = time.time()
        cpu_result = np.dot(a, b)
        cpu_time = time.time() - start
        
        # GPU测试
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        cp.cuda.Stream.null.synchronize()
        
        start = time.time()
        gpu_result = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        
        # 验证结果
        gpu_result_cpu = cp.asnumpy(gpu_result)
        max_diff = np.max(np.abs(cpu_result - gpu_result_cpu))
        
        speedup = cpu_time / gpu_time
        print(f"CPU时间: {cpu_time:.3f}s")
        print(f"GPU时间: {gpu_time:.3f}s")
        print(f"加速比: {speedup:.1f}x")
        print(f"精度误差: {max_diff:.2e}")
        
        if size == 4000 and speedup > 5:
            print("🎉 GPU性能优秀，适合大规模计算")
        elif speedup > 2:
            print("✅ GPU性能良好")
        else:
            print("⚠️ GPU加速效果有限")
    
    print("\n✅ GPU性能测试完成")
    
except ImportError:
    print("❌ CuPy未正确安装")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
'''
    
    # 将测试脚本写入临时文件
    with open('gpu_test.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    # 执行测试
    success, stdout, stderr = run_command(f"{sys.executable} gpu_test.py", timeout=120)
    
    # 清理临时文件
    try:
        os.remove('gpu_test.py')
    except:
        pass
    
    if success:
        print(stdout)
    else:
        print(f"测试执行失败: {stderr}")
    
    return success

def setup_optimization_environment():
    """设置优化算法环境"""
    print("\n⚙️ 配置优化算法环境...")
    
    # 检查优化脚本是否存在
    if os.path.exists('optimization_gpu.py'):
        print("✅ 找到GPU优化脚本: optimization_gpu.py")
    else:
        print("⚠️ 未找到optimization_gpu.py，请确保文件存在")
    
    # 创建运行脚本
    run_script = '''
#!/usr/bin/env python3
# 云端GPU优化算法运行脚本

import os
import sys
import time

print("🚀 启动GPU加速优化算法...")
print(f"Python版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")

# 检查GPU状态
os.system("nvidia-smi")

print("\n" + "="*50)
print("开始执行优化算法")
print("="*50)

start_time = time.time()

# 运行优化算法
if os.path.exists('optimization_gpu.py'):
    os.system(f"{sys.executable} optimization_gpu.py")
else:
    print("❌ 未找到optimization_gpu.py文件")
    sys.exit(1)

end_time = time.time()
print(f"\n总运行时间: {end_time - start_time:.2f} 秒")
'''
    
    with open('run_optimization.py', 'w', encoding='utf-8') as f:
        f.write(run_script)
    
    print("✅ 创建运行脚本: run_optimization.py")
    
    # 设置环境变量
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',  # 使用第一个GPU
        'CUPY_CACHE_DIR': '/tmp/cupy_cache',  # 设置CuPy缓存目录
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量: {key}={value}")

def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("🎯 云端GPU环境配置完成")
    print("="*60)
    
    print("\n📋 使用说明:")
    print("1. 运行优化算法:")
    print("   python run_optimization.py")
    print("   或")
    print("   python optimization_gpu.py")
    
    print("\n2. 监控GPU使用情况:")
    print("   watch -n 1 nvidia-smi")
    
    print("\n3. 检查GPU状态:")
    print("   nvidia-smi")
    
    print("\n4. 如果遇到内存不足:")
    print("   - 减少种群大小 (population_size)")
    print("   - 减少采样点数量")
    print("   - 使用更小的时间步长")
    
    print("\n⚠️ 注意事项:")
    print("- 确保选择了GPU实例类型")
    print("- 长时间运行可能产生费用")
    print("- 建议定期保存中间结果")
    print("- 可以使用Ctrl+C中断运行")
    
    print("\n🔗 相关文件:")
    print("- optimization_gpu.py: GPU加速优化算法")
    print("- run_optimization.py: 快速运行脚本")
    print("- install_gpu_deps.py: 依赖安装脚本")

def main():
    """主函数"""
    print("🌟 云端GPU环境快速配置脚本")
    print("=" * 50)
    
    start_time = time.time()
    
    # 步骤1: 检查GPU环境
    if not check_gpu_environment():
        print("\n❌ GPU环境检查失败，请检查实例配置")
        return
    
    # 步骤2: 安装依赖
    install_dependencies()
    
    # 步骤3: 安装CuPy
    if not install_cupy():
        print("\n❌ CuPy安装失败，将无法使用GPU加速")
        return
    
    # 步骤4: 测试GPU性能
    test_gpu_performance()
    
    # 步骤5: 配置环境
    setup_optimization_environment()
    
    # 步骤6: 显示使用说明
    print_usage_instructions()
    
    end_time = time.time()
    print(f"\n⏱️ 配置完成，耗时: {end_time - start_time:.1f} 秒")
    print("\n🚀 现在可以运行: python run_optimization.py")

if __name__ == "__main__":
    main()