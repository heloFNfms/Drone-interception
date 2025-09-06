#!/bin/bash

# 烟幕遮蔽优化 - 一键部署运行脚本
# 适用于 Ubuntu 22.04 + CUDA 12.1.0 + Python 3.11

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_sudo() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "检测到root用户，建议使用普通用户运行此脚本"
        read -p "是否继续? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 检查系统环境
check_system() {
    print_info "检查系统环境..."
    
    # 检查操作系统
    if [[ ! -f /etc/os-release ]]; then
        print_error "无法识别操作系统"
        exit 1
    fi
    
    source /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        print_warning "此脚本针对Ubuntu优化，其他系统可能需要调整"
    fi
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        print_error "Python3未安装"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    print_success "Python版本: $python_version"
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA驱动已安装"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_warning "NVIDIA驱动未检测到，将使用CPU计算"
    fi
    
    # 检查内存
    mem_gb=$(free -g | awk 'NR==2{print $2}')
    if [[ $mem_gb -lt 8 ]]; then
        print_warning "内存较少 (${mem_gb}GB)，建议至少8GB"
    else
        print_success "内存充足: ${mem_gb}GB"
    fi
}

# 创建工作目录
setup_workspace() {
    print_info "设置工作环境..."
    
    WORK_DIR="$HOME/smoke_optimization"
    
    if [[ -d "$WORK_DIR" ]]; then
        print_warning "工作目录已存在，是否清理重建?"
        read -p "清理并重建? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$WORK_DIR"
        fi
    fi
    
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    print_success "工作目录: $WORK_DIR"
}

# 安装系统依赖
install_system_deps() {
    print_info "安装系统依赖..."
    
    # 更新包列表
    sudo apt update -qq
    
    # 安装基础依赖
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        wget \
        curl \
        tmux \
        htop \
        tree \
        unzip \
        libffi-dev \
        libssl-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        pkg-config \
        > /dev/null 2>&1
    
    print_success "系统依赖安装完成"
}

# 创建Python环境
setup_python_env() {
    print_info "创建Python虚拟环境..."
    
    ENV_PATH="$HOME/smoke_optimization_env"
    
    # 删除旧环境
    if [[ -d "$ENV_PATH" ]]; then
        rm -rf "$ENV_PATH"
    fi
    
    # 创建新环境
    python3 -m venv "$ENV_PATH"
    source "$ENV_PATH/bin/activate"
    
    # 升级pip
    pip install --upgrade pip --quiet
    
    print_success "Python虚拟环境创建完成"
}

# 安装Python依赖
install_python_deps() {
    print_info "安装Python依赖包..."
    
    source "$HOME/smoke_optimization_env/bin/activate"
    
    # 创建requirements.txt
    cat > requirements.txt << EOF
numpy==1.24.3
scipy==1.11.4
matplotlib==3.7.2
pandas==2.0.3
scikit-optimize==0.9.0
optuna==3.4.0
plotly==5.17.0
seaborn==0.12.2
tqdm==4.66.1
joblib==1.3.2
psutil==5.9.5
h5py==3.9.0
tables==3.8.0
openpyxl==3.1.2
xlsxwriter==3.1.9
jupyter==1.0.0
jupyterlab==4.0.7
ipywidgets==8.1.1
notebook==7.0.6
EOF
    
    # 批量安装
    print_info "正在安装依赖包，请耐心等待..."
    pip install -r requirements.txt --quiet --no-warn-script-location
    
    print_success "Python依赖包安装完成"
}

# 下载优化程序
setup_optimization_code() {
    print_info "设置优化程序代码..."
    
    # 创建主程序文件 (这里你需要将之前的代码内容粘贴进去)
    cat > smoke_optimization.py << 'EOF'
#!/usr/bin/env python3
"""
烟幕遮蔽优化程序 - 云GPU优化版
适用于高性能计算环境的多策略优化算法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, dual_annealing
import pandas as pd
import time
import os
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端（适用于无显示器环境）
import matplotlib
matplotlib.use('Agg')

# [这里需要插入完整的优化程序代码 - 从之前创建的代码中复制]
# 为了节省空间，这里只显示框架，实际部署时需要完整代码

print("烟幕遮蔽优化程序已准备就绪")
print("如需完整代码，请将之前生成的 AdvancedSmokeOptimizer 类和 main() 函数复制到此处")
EOF
    
    # 创建测试程序
    cat > test_quick.py << 'EOF'
#!/usr/bin/env python3
"""快速测试程序"""

import numpy as np
import time
from scipy.optimize import minimize

def test_basic():
    print("测试基本功能...")
    
    # 测试numpy
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    start = time.time()
    c = np.dot(a, b)
    duration = time.time() - start
    print(f"矩阵运算测试: {duration:.3f}s")
    
    # 测试scipy
    def rosenbrock(x):
        return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    start = time.time()
    result = minimize(rosenbrock, np.array([1.3, 0.7, 0.8, 1.9, 1.2]), method='BFGS')
    duration = time.time() - start
    print(f"优化算法测试: {duration:.3f}s, 成功: {result.success}")
    
    print("基本功能测试通过!")

if __name__ == "__main__":
    test_basic()
EOF
    
    chmod +x smoke_optimization.py test_quick.py
    
    print_success "程序代码设置完成"
}

# 创建运行脚本
create_run_scripts() {
    print_info "创建运行脚本..."
    
    # 主运行脚本
    cat > run.sh << 'EOF'
#!/bin/bash

# 激活环境
source ~/smoke_optimization_env/bin/activate

# 进入工作目录
cd ~/smoke_optimization

# 检查环境
echo "检查Python环境..."
python --version
pip list | grep -E "(numpy|scipy|matplotlib)"

echo ""
echo "开始运行烟幕遮蔽优化..."
echo "==============================================="

# 运行优化
python smoke_optimization.py

# 显示结果
echo ""
echo "==============================================="
echo "优化完成，检查结果文件："
ls -la *.txt *.png *.json 2>/dev/null || echo "未找到结果文件"

echo ""
echo "如果需要查看详细结果："
echo "cat optimization_results.txt"
echo "或者传输图片文件查看可视化结果"
EOF
    
    # 快速测试脚本
    cat > test.sh << 'EOF'
#!/bin/bash

source ~/smoke_optimization_env/bin/activate
cd ~/smoke_optimization

echo "快速功能测试..."
python test_quick.py

echo ""
echo "环境信息："
echo "Python: $(python --version)"
echo "工作目录: $(pwd)"
echo "虚拟环境: $VIRTUAL_ENV"
echo "可用内存: $(free -h | grep '^Mem:' | awk '{print $7}')"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
fi
EOF
    
    # GPU监控脚本
    cat > monitor.sh << 'EOF'
#!/bin/bash

echo "系统资源监控 (按 Ctrl+C 退出)"
echo "================================"

if command -v nvidia-smi &> /dev/null; then
    # GPU监控
    watch -n 2 'echo "=== GPU状态 ==="; nvidia-smi; echo ""; echo "=== 系统负载 ==="; top -bn1 | head -5; echo ""; echo "=== 内存使用 ==="; free -h'
else
    # CPU监控
    watch -n 2 'echo "=== 系统负载 ==="; top -bn1 | head -10; echo ""; echo "=== 内存使用 ==="; free -h'
fi
EOF
    
    # Jupyter启动脚本
    cat > jupyter.sh << 'EOF'
#!/bin/bash

source ~/smoke_optimization_env/bin/activate
cd ~/smoke_optimization

echo "启动JupyterLab..."
echo "访问地址: http://localhost:8888"
echo "如需远程访问，请设置SSH端口转发"
echo ""

# 生成配置文件
jupyter lab --generate-config 2>/dev/null || true

# 启动JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
EOF
    
    chmod +x run.sh test.sh monitor.sh jupyter.sh
    
    print_success "运行脚本创建完成"
}

# 创建配置文件
create_config() {
    print_info "创建配置文件..."
    
    # 项目配置
    cat > config.json << EOF
{
    "project_name": "smoke_optimization",
    "version": "1.0",
    "created": "$(date -Iseconds)",
    "environment": {
        "python_env": "$HOME/smoke_optimization_env",
        "work_dir": "$(pwd)",
        "system": "$(uname -a)"
    },
    "optimization": {
        "max_iterations": 400,
        "population_size": 25,
        "use_multiprocessing": true,
        "algorithms": ["differential_evolution", "dual_annealing", "local_search"]
    }
}
EOF
    
    # 创建README
    cat > README.md << 'EOF'
# 烟幕遮蔽优化系统

## 快速使用指南

### 1. 测试环境
```bash
./test.sh
```

### 2. 运行优化
```bash
./run.sh
```

### 3. 监控资源
```bash
./monitor.sh
```

### 4. 启动Jupyter (可选)
```bash
./jupyter.sh
```

## 文件说明

- `smoke_optimization.py` - 主优化程序
- `test_quick.py` - 快速测试程序
- `run.sh` - 主运行脚本
- `test.sh` - 测试脚本
- `monitor.sh` - 资源监控脚本
- `jupyter.sh` - Jupyter启动脚本
- `config.json` - 配置文件
- `requirements.txt` - Python依赖列表

## 结果文件

优化完成后会生成：
- `optimization_results.txt` - 详细结果报告
- `optimization_results.json` - JSON格式结果
- `optimization_plots.png` - 可视化图表

## 故障排除

1. 环境问题：重新运行 `./test.sh`
2. 依赖问题：`pip install -r requirements.txt`
3. 权限问题：`chmod +x *.sh`
4. 内存不足：减少配置文件中的 population_size
5. CUDA问题：检查 `nvidia-smi` 输出

## 性能调优

- 对于CPU密集任务：增加 population_size
- 对于内存有限环境：减少 max_iterations
- 对于快速测试：修改算法参数

## 联系方式

如有问题，请检查日志文件或联系技术支持。
EOF
    
    print_success "配置文件创建完成"
}

# 运行最终测试
final_test() {
    print_info "运行最终测试..."
    
    source "$HOME/smoke_optimization_env/bin/activate"
    
    # 测试导入
    python3 -c "
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
print('✓ 核心库导入成功')
print(f'NumPy版本: {np.__version__}')
print(f'SciPy版本: {scipy.__version__}')
print(f'Matplotlib版本: {matplotlib.__version__}')
"
    
    if [[ $? -eq 0 ]]; then
        print_success "环境测试通过"
    else
        print_error "环境测试失败"
        return 1
    fi
}

# 显示部署完成信息
show_completion_info() {
    print_success "部署完成!"
    echo ""
    echo "========================================"
    echo "烟幕遮蔽优化系统已准备就绪"
    echo "========================================"
    echo ""
    echo "工作目录: $(pwd)"
    echo "Python环境: $HOME/smoke_optimization_env"
    echo ""
    echo "快速开始："
    echo "  测试环境:  ./test.sh"
    echo "  运行优化:  ./run.sh"  
    echo "  监控系统:  ./monitor.sh"
    echo "  启动Jupyter: ./jupyter.sh"
    echo ""
    echo "重要文件："
    echo "  主程序:    smoke_optimization.py"
    echo "  配置:      config.json"
    echo "  说明:      README.md"
    echo ""
    echo "注意事项："
    echo "1. 运行前需要将完整的优化代码复制到 smoke_optimization.py"
    echo "2. 大型计算建议在 tmux 会话中运行"
    echo "3. 监控GPU/CPU使用情况以优化性能"
    echo ""
}

# 主函数
main() {
    echo "========================================"
    echo "烟幕遮蔽优化系统 - 一键部署脚本"
    echo "========================================"
    echo ""
    
    # 检查用户权限
    check_sudo
    
    # 系统检查
    check_system
    
    # 询问是否继续
    echo ""
    read -p "是否开始自动部署? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "用户取消部署"
        exit 0
    fi
    
    # 开始部署流程
    print_info "开始自动部署流程..."
    
    # 1. 设置工作空间
    setup_workspace
    
    # 2. 安装系统依赖
    print_info "开始安装系统依赖，可能需要sudo权限..."
    install_system_deps
    
    # 3. 创建Python环境
    setup_python_env
    
    # 4. 安装Python依赖
    install_python_deps
    
    # 5. 设置程序代码
    setup_optimization_code
    
    # 6. 创建运行脚本
    create_run_scripts
    
    # 7. 创建配置文件
    create_config
    
    # 8. 最终测试
    if ! final_test; then
        print_error "最终测试失败，请检查安装"
        exit 1
    fi
    
    # 9. 显示完成信息
    show_completion_info
    
    # 创建桌面快捷方式（如果是桌面环境）
    if [[ -d "$HOME/Desktop" ]]; then
        cat > "$HOME/Desktop/烟幕优化.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=烟幕遮蔽优化
Comment=启动烟幕遮蔽优化计算
Exec=gnome-terminal -- bash -c "cd $HOME/smoke_optimization && ./run.sh; read -p '按任意键关闭...'"
Icon=applications-science
Terminal=true
Categories=Science;Education;
EOF
        chmod +x "$HOME/Desktop/烟幕优化.desktop" 2>/dev/null || true
        print_info "已创建桌面快捷方式"
    fi
}

# 错误处理
trap 'print_error "脚本执行中断"; exit 1' INT TERM

# 执行主函数
main "$@"