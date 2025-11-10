#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rl/TankEnv.h"

namespace py = pybind11;
using namespace TankTrouble;

PYBIND11_MODULE(tank_trouble_env, m)
{
    m.doc() = "TankTrouble RL environment bindings";

    py::class_<RLController>(m, "RLController")
        .def(py::init<LocalController *>(), py::arg("shared") = nullptr);

    py::class_<TankEnv>(m, "TankEnv")
        .def(py::init<>())
        .def(py::init<RLController *>(), py::arg("rl_controller") = nullptr)
        .def("reset", &TankEnv::reset)
        .def("step", &TankEnv::step)
        .def("get_smith_action", &TankEnv::getSmithAction)
        .def("get_agent_smith_action", &TankEnv::getAgentSmithAction);

    py::enum_<TankEnv::Action>(m, "Action")
        .value("DO_NOTHING", TankEnv::Action::DO_NOTHING)
        .value("MOVE_FORWARD", TankEnv::Action::MOVE_FORWARD)
        .value("MOVE_BACKWARD", TankEnv::Action::MOVE_BACKWARD)
        .value("ROTATE_CW", TankEnv::Action::ROTATE_CW)
        .value("ROTATE_CCW", TankEnv::Action::ROTATE_CCW)
        .value("SHOOT", TankEnv::Action::SHOOT);
}
