// Pybind11 bindings for RLController
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "controller/RLController.h"

namespace py = pybind11;
using namespace TankTrouble;

PYBIND11_MODULE(rl_controller, m)
{
    m.doc() = "RL Controller bindings for GUI training";

    // Expose RLController as standalone type (no base class exposed to Python)
    py::class_<RLController>(m, "RLController")
        .def(py::init<>())
        .def("setGetActionCallback", &RLController::setGetActionCallback)
        .def("setEpisodeEndCallback", &RLController::setEpisodeEndCallback)
        // .def("get_smith_action", &RLController::getSmithAction) // Removed: method no longer exists
        .def("getEpisodeCount", &RLController::getEpisodeCount)
        .def("getTotalReward", &RLController::getTotalReward)
        .def("start", &RLController::start)
        .def("quitGame", &RLController::quitGame);

    // Expose a helper to access the current (global) RLController instance created by the app
    m.def("get_global_controller", []() -> RLController *
          { return RLController::getGlobalInstance(); }, py::return_value_policy::reference);
}
