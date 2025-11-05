#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rl/TankEnv.h"

namespace py = pybind11;
using namespace TankTrouble;

PYBIND11_MODULE(tank_trouble_env, m)
{
    m.doc() = "TankTrouble RL environment bindings";

    py::class_<TankEnv>(m, "TankEnv")
        .def(py::init<>())
        .def("reset", &TankEnv::reset)
        .def("step", &TankEnv::step)
        .def("set_reward_config", [](TankEnv& env, py::dict d){
            std::unordered_map<std::string, double> m;
            for(auto item : d)
                m[item.first.cast<std::string>()] = item.second.cast<double>();
            env.setRewardConfig(m);
        });

    py::enum_<TankEnv::Action>(m, "Action")
        .value("DO_NOTHING", TankEnv::Action::DO_NOTHING)
        .value("MOVE_FORWARD", TankEnv::Action::MOVE_FORWARD)
        .value("MOVE_BACKWARD", TankEnv::Action::MOVE_BACKWARD)
        .value("ROTATE_CW", TankEnv::Action::ROTATE_CW)
        .value("ROTATE_CCW", TankEnv::Action::ROTATE_CCW)
        .value("SHOOT", TankEnv::Action::SHOOT);
}


