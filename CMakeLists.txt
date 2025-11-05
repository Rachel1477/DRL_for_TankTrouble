cmake_minimum_required(VERSION 3.8)
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
