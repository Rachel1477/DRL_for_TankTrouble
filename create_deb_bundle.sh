#!/bin/bash
set -e

# ============================================
# TankTrouble .deb 打包脚本（包含所有依赖）
# ============================================

PACKAGE_NAME="tanktrouble"
VERSION="1.0.0"
ARCH="amd64"
BUILD_DIR="build"
DEB_DIR="${PACKAGE_NAME}_${VERSION}_${ARCH}"

echo "=========================================="
echo "创建 TankTrouble .deb 包（包含依赖库）"
echo "=========================================="

# 检查编译是否完成
if [ ! -f "${BUILD_DIR}/TankTrouble" ]; then
    echo "错误: 未找到编译好的可执行文件"
    echo "请先运行: bash build.sh"
    exit 1
fi

# 清理旧包
rm -rf "${DEB_DIR}"
rm -f "${DEB_DIR}.deb"

# 创建目录结构
echo "==> 创建目录结构..."
mkdir -p "${DEB_DIR}/DEBIAN"
mkdir -p "${DEB_DIR}/opt/${PACKAGE_NAME}/bin"
mkdir -p "${DEB_DIR}/opt/${PACKAGE_NAME}/lib"
mkdir -p "${DEB_DIR}/opt/${PACKAGE_NAME}/python"
mkdir -p "${DEB_DIR}/usr/bin"
mkdir -p "${DEB_DIR}/usr/share/applications"
mkdir -p "${DEB_DIR}/usr/share/doc/${PACKAGE_NAME}"

# 复制主程序
echo "==> 复制主程序..."
cp "${BUILD_DIR}/TankTrouble" "${DEB_DIR}/opt/${PACKAGE_NAME}/bin/"
chmod +x "${DEB_DIR}/opt/${PACKAGE_NAME}/bin/TankTrouble"

# 复制资源文件
echo "==> 复制资源文件..."
cp "${BUILD_DIR}/entry.jpg" "${DEB_DIR}/opt/${PACKAGE_NAME}/" 2>/dev/null || true
cp README.md "${DEB_DIR}/usr/share/doc/${PACKAGE_NAME}/"
cp LICENSE "${DEB_DIR}/usr/share/doc/${PACKAGE_NAME}/" 2>/dev/null || true

# 复制 Python 模块（如果存在）
echo "==> 复制 Python 模块..."
if [ -f "${BUILD_DIR}/tank_trouble_env.so" ]; then
    cp "${BUILD_DIR}/tank_trouble_env.so" "${DEB_DIR}/opt/${PACKAGE_NAME}/python/"
fi
if [ -f "${BUILD_DIR}/rl_controller.so" ]; then
    cp "${BUILD_DIR}/rl_controller.so" "${DEB_DIR}/opt/${PACKAGE_NAME}/python/"
fi

# 复制 Python 训练脚本
if [ -d "python" ]; then
    cp -r python/*.py "${DEB_DIR}/opt/${PACKAGE_NAME}/python/" 2>/dev/null || true
fi

# 收集依赖的共享库
echo "==> 收集依赖的共享库..."
collect_dependencies() {
    local binary=$1
    local lib_dir=$2
    
    # 获取所有非系统库的依赖
    ldd "$binary" | grep "=>" | while read -r line; do
        lib_path=$(echo "$line" | awk '{print $3}')
        
        # 跳过空路径
        if [ -z "$lib_path" ]; then
            continue
        fi
        
        # 只复制 conda 环境中的库和非标准系统库
        if [[ "$lib_path" == *"miniconda"* ]] || [[ "$lib_path" == *"conda"* ]]; then
            lib_name=$(basename "$lib_path")
            if [ ! -f "${lib_dir}/${lib_name}" ]; then
                echo "    收集: $lib_name"
                cp "$lib_path" "${lib_dir}/"
                # 递归收集这个库的依赖
                collect_dependencies "$lib_path" "$lib_dir"
            fi
        fi
    done
}

collect_dependencies "${BUILD_DIR}/TankTrouble" "${DEB_DIR}/opt/${PACKAGE_NAME}/lib"

# 创建 control 文件
echo "==> 创建 control 文件..."
cat > "${DEB_DIR}/DEBIAN/control" << EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: games
Priority: optional
Architecture: ${ARCH}
Depends: libgtkmm-3.0-1v5, libatkmm-1.6-1v5, libglibmm-2.4-1v5, libcairomm-1.0-1v5, libpangomm-1.4-1v5
Recommends: python3, python3-pip
Installed-Size: $(du -sk "${DEB_DIR}" | cut -f1)
Maintainer: Rachel <rachel@example.com>
Homepage: https://github.com/your-repo/DRL_for_TankTrouble
Description: TankTrouble - Deep Reinforcement Learning Tank Battle Game
 TankTrouble is a tank battle game with integrated deep reinforcement
 learning training capabilities.
 .
 This package includes all necessary dependencies including Python libraries.
 .
 Features:
  * Single-player mode with intelligent AI (Agent Smith)
  * RL training environment with Gym-like interface
  * GUI training visualization
  * Random maze generation
  * Physics engine with collision detection
EOF

# 创建启动脚本
echo "==> 创建启动脚本..."
cat > "${DEB_DIR}/usr/bin/${PACKAGE_NAME}" << 'EOF'
#!/bin/bash

# TankTrouble 启动脚本
INSTALL_DIR="/opt/tanktrouble"

# 检查安装目录
if [ ! -d "${INSTALL_DIR}" ]; then
    echo "错误: 安装目录不存在: ${INSTALL_DIR}"
    exit 1
fi

# 检查可执行文件
if [ ! -x "${INSTALL_DIR}/bin/TankTrouble" ]; then
    echo "错误: 可执行文件不存在或无执行权限: ${INSTALL_DIR}/bin/TankTrouble"
    exit 1
fi

# 设置库路径（优先使用打包的库）
export LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"

# 设置 Python 环境（如果需要 RL 功能）
export PYTHONPATH="${INSTALL_DIR}/python:${PYTHONPATH}"
export PYTHONHOME="${INSTALL_DIR}/lib"

# 调试模式（取消注释以启用）
# export LD_DEBUG=libs

# 切换到安装目录（资源文件如 entry.jpg 在这里）
cd "${INSTALL_DIR}" || {
    echo "错误: 无法切换到安装目录"
    exit 1
}

# 运行程序
exec "${INSTALL_DIR}/bin/TankTrouble" "$@" 2>&1
EOF
chmod +x "${DEB_DIR}/usr/bin/${PACKAGE_NAME}"

# 创建桌面快捷方式
echo "==> 创建桌面快捷方式..."
cat > "${DEB_DIR}/usr/share/applications/${PACKAGE_NAME}.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=TankTrouble
Comment=Tank Battle Game with Deep Reinforcement Learning
Exec=${PACKAGE_NAME}
Icon=applications-games
Terminal=false
Categories=Game;ArcadeGame;
Keywords=tank;game;ai;reinforcement-learning;
StartupNotify=true
EOF

# 创建 postinst 脚本
echo "==> 创建 postinst 脚本..."
cat > "${DEB_DIR}/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e

# 更新桌面数据库
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  TankTrouble 安装成功！"
echo "============================================"
echo ""
echo "运行游戏："
echo "  终端输入: tanktrouble"
echo "  或从应用菜单启动"
echo ""
echo "注意："
echo "- 基本游戏功能（单机模式）可以直接使用"
echo "- 如需 RL 训练功能，请安装 Python 依赖："
echo "    pip3 install torch numpy pybind11 gymnasium"
echo ""
echo "============================================"

exit 0
EOF
chmod +x "${DEB_DIR}/DEBIAN/postinst"

# 创建 prerm 脚本
cat > "${DEB_DIR}/DEBIAN/prerm" << 'EOF'
#!/bin/bash
set -e
exit 0
EOF
chmod +x "${DEB_DIR}/DEBIAN/prerm"

# 设置权限
echo "==> 设置文件权限..."
find "${DEB_DIR}/opt/${PACKAGE_NAME}" -type f -name "*.so*" -exec chmod 755 {} \;
find "${DEB_DIR}" -type d -exec chmod 755 {} \;

# 修复 RPATH（使用相对路径）
echo "==> 修复 RPATH..."
if command -v patchelf >/dev/null 2>&1; then
    patchelf --set-rpath '$ORIGIN/../lib' "${DEB_DIR}/opt/${PACKAGE_NAME}/bin/TankTrouble" 2>/dev/null || true
    echo "    RPATH 已设置为相对路径"
else
    echo "    警告: 未找到 patchelf，跳过 RPATH 设置"
    echo "    安装命令: sudo apt-get install patchelf"
fi

# 构建 .deb 包
echo "==> 构建 .deb 包..."
fakeroot dpkg-deb --build "${DEB_DIR}"

# 显示包信息
echo ""
echo "============================================"
echo "  ✓ .deb 包创建成功！"
echo "============================================"
echo ""
echo "包文件: ${DEB_DIR}.deb"
echo "包大小: $(du -h "${DEB_DIR}.deb" | cut -f1)"
echo ""
echo "安装命令:"
echo "  sudo dpkg -i ${DEB_DIR}.deb"
echo "  sudo apt-get install -f  # 自动修复依赖"
echo ""
echo "卸载命令:"
echo "  sudo apt remove ${PACKAGE_NAME}"
echo ""
echo "查看包内容:"
echo "  dpkg -c ${DEB_DIR}.deb"
echo ""
echo "============================================"

