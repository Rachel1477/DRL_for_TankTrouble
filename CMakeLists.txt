cmake_minimum_required(VERSION 3.18)
project(TankTrouble)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(TANK_TROUBLE_SRC main.cc
        Tank.cc
        Shell.cc
        Block.cc
        Object.cc
        util/Vec.cc
        util/Math.cc
        util/Id.cc
        Maze.cc
        Controller.cc
        controller/LocalController.cc
        controller/RLController.cc
        event/ControlEvent.cc
        smithAI/AgentSmith.cc
        smithAI/DodgeStrategy.cc
        smithAI/AStar.cc
        smithAI/ContactStrategy.cc
        smithAI/AttackStrategy.cc
        view/EntryView.cc
        view/GameView.cc
        view/component/GameArea.cc
        view/component/PlayerInfoItem.cc
        Window.cc)

find_package(PkgConfig)
pkg_check_modules(GTKMM gtkmm-3.0)

include_directories(${GTKMM_INCLUDE_DIRS} .)
link_directories(${GTKMM_LIBRARY_DIRS})
link_libraries(${GTKMM_LIBRARIES} pthread)

file(COPY ${CMAKE_SOURCE_DIR}/res/entry.jpg DESTINATION ${CMAKE_BINARY_DIR})
add_executable(TankTrouble ${TANK_TROUBLE_SRC})

# --- 从这里开始修改 ---

# 找到 Conda 环境 "RL" 中的 Python
# 优先用用户手动配置的路径，否则尝试从 PATH 中查找
if(NOT CONDA_EXECUTABLE)
    find_program(CONDA_EXECUTABLE conda)
endif()
if(NOT CONDA_EXECUTABLE)
    # 兼容常见 miniconda 路径（如用户提供的路径保持生效）
    set(CONDA_EXECUTABLE /home/rachel/miniconda3/condabin/conda)
endif()
if(NOT CONDA_EXECUTABLE)
    message(FATAL_ERROR "Conda executable not found. Please make sure Conda is in your PATH.")
endif()

execute_process(
        COMMAND ${CONDA_EXECUTABLE} run -n RL --no-capture-output python -c "import sys; print(sys.executable)"
        OUTPUT_VARIABLE PYTHON_EXECUTABLE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CONDA_PYTHON_RESULT
)

if(CONDA_PYTHON_RESULT)
    message(FATAL_ERROR "Failed to find Python executable in conda environment 'RL'. Make sure the environment exists and Python is installed.")
else()
    message(STATUS "Found Python in conda env 'RL': ${PYTHON_EXECUTABLE_PATH}")
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE_PATH})
endif()

# 读取 Python 前缀与版本号（用于运行时 PYTHONHOME/PYTHONPATH 设置）
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sys; print(sys.prefix)"
        OUTPUT_VARIABLE CONDA_PY_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
        OUTPUT_VARIABLE CONDA_PY_VER
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Conda Python prefix: ${CONDA_PY_PREFIX}")
message(STATUS "Conda Python version: ${CONDA_PY_VER}")

# 使用找到的 Python 来查找 pybind11（来自该环境）
find_package(Python REQUIRED)
# 强制 Python3 指向 conda 环境，用于链接嵌入式解释器
unset(Python3_FOUND CACHE)
list(APPEND CMAKE_PREFIX_PATH "${CONDA_PY_PREFIX}")
find_package(Python3 COMPONENTS Interpreter Development Development.Module Development.Embed REQUIRED)
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_LOCATE_RESULT
)

if(PYBIND11_LOCATE_RESULT)
    message(WARNING "pybind11 not found in conda environment 'RL'. Install with: conda activate RL && pip install pybind11")
    set(pybind11_FOUND FALSE)
else()
    message(STATUS "Found pybind11 cmake dir: ${pybind11_DIR}")
    find_package(pybind11 REQUIRED)
endif()

# Link pybind11 embed for Python callbacks in Window (if available)
if(pybind11_FOUND)
    target_link_libraries(TankTrouble PRIVATE pybind11::embed)
    target_compile_definitions(TankTrouble PRIVATE
        HAVE_PYBIND11=1
        PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
        PROJECT_BUILD_DIR="${CMAKE_BINARY_DIR}"
        CONDA_PY_PREFIX="${CONDA_PY_PREFIX}"
        CONDA_PY_VER="${CONDA_PY_VER}"
    )
    target_include_directories(TankTrouble PRIVATE ${Python3_INCLUDE_DIRS})
    # 链接嵌入式 Python。优先目标，其次直接库变量
    if (TARGET Python3::Embed)
        target_link_libraries(TankTrouble PRIVATE Python3::Embed)
    elseif (TARGET Python3::Python)
        target_link_libraries(TankTrouble PRIVATE Python3::Python)
    endif()
    if (Python3_LIBRARIES)
        target_link_libraries(TankTrouble PRIVATE ${Python3_LIBRARIES})
    endif()
    if (Python3_LIBRARY_DIRS)
        # 运行时rpath指向conda的lib目录
        foreach(_p ${Python3_LIBRARY_DIRS})
            target_link_options(TankTrouble PRIVATE -Wl,-rpath,${_p})
        endforeach()
    else()
        target_link_options(TankTrouble PRIVATE -Wl,-rpath,${CONDA_PY_PREFIX}/lib)
    endif()
    if (TARGET pybind11::headers)
        target_link_libraries(TankTrouble PRIVATE pybind11::headers)
    endif()
endif()

# Pybind11 module for RL environment
if(pybind11_FOUND)
    message(STATUS "pybind11 found, building Python module")

    # Source files needed for TankEnv (without GUI dependencies)
    set(RL_ENV_SRC
            rl/TankEnv.cc
            Tank.cc
            Shell.cc
            Block.cc
            Object.cc
            util/Vec.cc
            util/Math.cc
            util/Id.cc
            Maze.cc
            Controller.cc
            controller/LocalController.cc
            controller/RLController.cc
            event/ControlEvent.cc
            smithAI/AgentSmith.cc
            smithAI/DodgeStrategy.cc
            smithAI/AStar.cc
            smithAI/ContactStrategy.cc
            smithAI/AttackStrategy.cc
            view/component/GameArea.cc
            bindings/bindings.cc
    )

    if (COMMAND python_add_library)
        pybind11_add_module(tank_trouble_env ${RL_ENV_SRC})
    else()
        message(WARNING "python_add_library not available; falling back to manual MODULE target for tank_trouble_env")
        add_library(tank_trouble_env MODULE ${RL_ENV_SRC})
        set_target_properties(tank_trouble_env PROPERTIES PREFIX "" OUTPUT_NAME "tank_trouble_env")
        target_compile_definitions(tank_trouble_env PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)
        target_include_directories(tank_trouble_env PRIVATE ${GTKMM_INCLUDE_DIRS} .)
        target_link_libraries(tank_trouble_env PRIVATE pthread Python3::Module pybind11::headers)
    endif()

    # Build rl_controller module for RLController callbacks
    set(RL_CONTROLLER_SRC
            bindings/rl_bindings.cc
            Tank.cc
            Shell.cc
            Block.cc
            Object.cc
            util/Vec.cc
            util/Math.cc
            util/Id.cc
            Maze.cc
            Controller.cc
            controller/LocalController.cc
            controller/RLController.cc
            event/ControlEvent.cc
            smithAI/AgentSmith.cc
            smithAI/DodgeStrategy.cc
            smithAI/AStar.cc
            smithAI/ContactStrategy.cc
            smithAI/AttackStrategy.cc
            view/component/GameArea.cc
    )
    if (COMMAND python_add_library)
        pybind11_add_module(rl_controller ${RL_CONTROLLER_SRC})
    else()
        message(WARNING "python_add_library not available; falling back to manual MODULE target for rl_controller")
        add_library(rl_controller MODULE ${RL_CONTROLLER_SRC})
        set_target_properties(rl_controller PROPERTIES PREFIX "" OUTPUT_NAME "rl_controller")
        target_compile_definitions(rl_controller PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)
        target_include_directories(rl_controller PRIVATE ${GTKMM_INCLUDE_DIRS} .)
        target_link_libraries(rl_controller PRIVATE pthread Python3::Module pybind11::headers)
    endif()
else()
    message(WARNING "pybind11 not found. Install with: conda activate RL && pip install pybind11 or use git submodule")
endif()