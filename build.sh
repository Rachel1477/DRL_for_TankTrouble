conda activate RL
cd /home/visier/CLionProjects/DRL_for_TankTrouble
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/home/visier/miniconda3/envs/RL/bin/python -DPython3_ROOT_DIR=/home/visier/miniconda3/envs/RL -DCMAKE_PREFIX_PATH=/home/visier/miniconda3/envs/RL ..
make -j$(nproc)
