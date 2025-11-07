conda activate RL
cd /home/rachel/CLionProjects/DRL_for_TankTrouble
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/home/rachel/miniconda3/envs/RL/bin/python -DPython3_ROOT_DIR=/home/rachel/miniconda3/envs/RL -DCMAKE_PREFIX_PATH=/home/rachel/miniconda3/envs/RL ..
make -j$(nproc)
