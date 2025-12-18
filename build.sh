#!/bin/bash
# TankTrouble 编译脚本
# 使用 conda RL 环境进行编译，避免 libstdc++.so.6 版本冲突

set -e  # 遇到错误立即退出

echo "========================================"
echo "TankTrouble GUI RL Training 编译脚本"
echo "========================================"

# 检查是否在项目根目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 激活 conda RL 环境
echo ""
echo "步骤 1: 激活 conda RL 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RL

# 获取 conda 环境路径
CONDA_PREFIX=$(conda info --base)/envs/RL
echo "Conda RL 环境路径: $CONDA_PREFIX"

# 检查 Python 是否存在
if [ ! -f "$CONDA_PREFIX/bin/python" ]; then
    echo "错误: 未找到 Python 解释器，请确保 RL 环境已正确创建"
    echo "创建环境: conda create -n RL python=3.10"
    exit 1
fi

# 检查 pybind11
echo ""
echo "步骤 2: 检查依赖..."
if ! python -c "import pybind11" 2>/dev/null; then
    echo "未找到 pybind11，正在安装..."
    pip install pybind11
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "未找到 PyTorch，正在安装..."
    pip install torch numpy
fi

echo "✓ 依赖检查完成"

# 清理并创建 build 目录
echo ""
echo "步骤 3: 清理旧的 build 目录..."
rm -rf build
mkdir build
cd build

# 配置 CMake（关键：指定 Python 路径和 RPATH）
echo ""
echo "步骤 4: 配置 CMake..."
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DPython3_ROOT_DIR=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_INSTALL_RPATH=$CONDA_PREFIX/lib \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    ..

if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败"
    exit 1
fi

# 编译
echo ""
echo "步骤 5: 编译项目..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ 编译成功！"
echo "========================================"
echo ""
echo "运行方式："
echo "  cd build"
echo "  ./TankTrouble"
echo ""
echo "然后点击 'Agent训练' 按钮开始训练"
echo ""
