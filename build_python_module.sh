#!/bin/bash
# Build script for Python module with pybind11

set -e

echo "Building TankTrouble Python module..."

# Check if pybind11 is installed
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "Error: pybind11 not found. Installing..."
    pip install pybind11
fi

# Create build directory
mkdir -p build
cd build

# Configure and build
echo "Configuring CMake..."
cmake ..

echo "Building..."
make -j$(nproc)

# Check if module was built
if [ -f "tank_trouble_env*.so" ] || [ -f "tank_trouble_env.cpython*.so" ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "To use the module, set PYTHONPATH:"
    echo "  export PYTHONPATH=\$PWD/build:\$PYTHONPATH"
    echo ""
    echo "Or run from project root:"
    echo "  PYTHONPATH=./build python3 python/test_env.py"
else
    echo "Warning: Module file not found. Check build output for errors."
    exit 1
fi

