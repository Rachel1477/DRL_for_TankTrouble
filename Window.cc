//
// Created by zr on 23-2-16.
//

#include "Window.h"

#include <memory>
#include "view/GameView.h"
#include "defs.h"
#include "event/ControlEvent.h"
#include "controller/LocalController.h"
#include "controller/RLController.h"
#include <cstdlib>
#ifdef HAVE_PYBIND11
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#endif

namespace TankTrouble
{
    Window::Window():
        ctl(nullptr),
        KeyUpPressed(false), KeyDownPressed(false),
        KeyLeftPressed(false), KeyRightPressed(false),
        spacePressed(false)
    {
        set_title("TankTrouble");
        set_default_size(WINDOW_WIDTH, WINDOW_HEIGHT);
        set_resizable(false);
        add_events(Gdk::KEY_PRESS_MASK | Gdk::KEY_RELEASE_MASK);
        entryView.signal_choose_local().connect(sigc::mem_fun(*this, &Window::onUserChooseLocal));
        entryView.signal_choose_rl_training().connect(sigc::mem_fun(*this, &Window::onUserChooseRLTraining));
        add(entryView);

        loginSuccessNotifier.connect(sigc::mem_fun(*this, &Window::onLoginSuccess));
        roomUpdateNotifier.connect(sigc::mem_fun(*this, &Window::onRoomsUpdate));
        gameOnNotifier.connect(sigc::mem_fun(*this, &Window::onGameBegin));
        gameOffNotifier.connect(sigc::mem_fun(*this, &Window::onGameOff));

        entryView.show();
    }

    void Window::notifyLoginSuccess() {loginSuccessNotifier.emit();}

    void Window::notifyRoomUpdate() {roomUpdateNotifier.emit();}

    void Window::notifyGameOn() {gameOnNotifier.emit();}

    void Window::notifyGameOff() {gameOffNotifier.emit();}

    void Window::onUserChooseLocal()
    {
        remove();
        ctl = std::make_unique<LocalController>();
        ctl->start();
        gameView = std::make_unique<GameView>(ctl.get());
        gameView->signal_quit_game().connect(sigc::mem_fun(*this, &Window::toEntryView));
        add(*gameView);
        gameView->show();
    }

    void Window::onUserChooseRLTraining()
    {
        remove();
        // Create RL controller for training
        auto* rlCtl = new RLController();
        ctl = std::unique_ptr<Controller>(rlCtl);

        // Inject Python DQN callbacks
        try
        {
#ifdef HAVE_PYBIND11
            static std::unique_ptr<pybind11::scoped_interpreter> guard;
            // 设置 CONDA 环境变量，确保嵌入式 Python 使用 conda 环境
            setenv("PYTHONHOME", CONDA_PY_PREFIX, 1);
            std::string site = std::string(CONDA_PY_PREFIX) + "/lib/python" + CONDA_PY_VER + "/site-packages";
            // 追加工程模块路径与 conda site-packages 到 PYTHONPATH
            const char* oldpp = getenv("PYTHONPATH");
            std::string newpp = site + ":" + std::string(PROJECT_BUILD_DIR) + ":" + std::string(PROJECT_ROOT_DIR) + "/build:" + std::string(PROJECT_ROOT_DIR) + "/cmake-build-debug:" + std::string(PROJECT_ROOT_DIR) + "/python";
            if(oldpp && *oldpp) newpp = newpp + ":" + std::string(oldpp);
            setenv("PYTHONPATH", newpp.c_str(), 1);

            if(!guard) guard = std::make_unique<pybind11::scoped_interpreter>();
            namespace py = pybind11;
            
            // Store ALL Python objects as static to keep them alive after GIL release
            static py::object sys_module;
            static py::object trainer_module;
            static py::object get_action_py;
            static py::object on_episode_end_py;
            
            if(!sys_module) {
                sys_module = py::module_::import("sys");
                py::list path = sys_module.attr("path");
                // Add absolute project paths to sys.path to avoid WD issues
                path.append(PROJECT_BUILD_DIR);
                path.append(PROJECT_ROOT_DIR "/build");
                path.append(PROJECT_ROOT_DIR "/cmake-build-debug");
                path.append(PROJECT_ROOT_DIR "/python");
            }

            if(!trainer_module) {
                trainer_module = py::module_::import("train_with_gui");
            }
            
            // initialize agent (state=57, action=6)
            try { trainer_module.attr("initialize_agent")(122, 6); } catch(...) {}  // Updated state size

            // Store callbacks
            static py::object on_step_py;
            get_action_py = trainer_module.attr("get_action_from_state");
            on_episode_end_py = trainer_module.attr("on_episode_end");
            on_step_py = trainer_module.attr("on_step");
            
            // Increment ref count to keep objects alive after GIL release
            get_action_py.inc_ref();
            on_episode_end_py.inc_ref();
            on_step_py.inc_ref();

            // Wrap in lambdas that acquire GIL
            auto get_action_cb = [](const std::vector<double>& state) -> int {
                py::gil_scoped_acquire acquire;
                static py::object& func = get_action_py;
                return func(state).cast<int>();
            };
            
            auto episode_end_cb = [](int episode, double reward, bool won) {
                py::gil_scoped_acquire acquire;
                static py::object& func = on_episode_end_py;
                func(episode, reward, won);
            };
            
            auto step_cb = [](const std::vector<double>& prev_state, int prev_action, 
                             double reward, const std::vector<double>& next_state, bool done) {
                py::gil_scoped_acquire acquire;
                static py::object& func = on_step_py;
                func(prev_state, prev_action, reward, next_state, done);
            };
            
            static_cast<RLController*>(ctl.get())->setGetActionCallback(get_action_cb);
            static_cast<RLController*>(ctl.get())->setEpisodeEndCallback(episode_end_cb);
            static_cast<RLController*>(ctl.get())->setStepCallback(step_cb);
            std::cout << "[RL] Python callbacks injected successfully (sys.path augmented)" << std::endl;
            
            // CRITICAL: Release GIL to allow agentLoop thread to acquire it
            // The scoped_interpreter holds GIL by default; we must explicitly release it
            // so that worker threads can call Python callbacks
            PyEval_SaveThread();
            std::cout << "[RL] GIL released for worker threads" << std::endl;
#endif
        }
        catch(const std::exception& e)
        {
            std::cerr << "[RL] Python callback injection failed: " << e.what() << std::endl;
        }

        // Start controller and show view
        ctl->start();
        gameView = std::make_unique<GameView>(ctl.get());
        gameView->signal_quit_game().connect(sigc::mem_fun(*this, &Window::toEntryView));
        add(*gameView);
        gameView->show();
    }

    

    void Window::toEntryView()
    {
        remove();
        if(gameView) gameView.reset();
        if(ctl) ctl.reset();
        add(entryView);
        entryView.show();
    }

    void Window::onLoginSuccess()
    {
        // no-op in local-only mode
    }

    void Window::onRoomsUpdate()
    {
        // no-op in local-only mode
    }

    void Window::onGameBegin()
    {
        remove();
        gameView = std::make_unique<GameView>(ctl.get());
        add(*gameView);
        gameView->signal_quit_game().
            connect(sigc::mem_fun(*this, &Window::onGameOff));
        gameView->show();
    }

    void Window::onGameOff()
    {
        // on quitting local game, return to entry view
        toEntryView();
    }

    bool Window::on_key_press_event(GdkEventKey* key_event)
    {
        if(!ctl)
            return Gtk::Window::on_key_press_event(key_event);
        // Disable human control in RL training mode
        if(dynamic_cast<RLController*>(ctl.get()) != nullptr)
            return Gtk::Window::on_key_press_event(key_event);
        if(!KeyUpPressed && key_event->keyval == GDK_KEY_Up)
        {
            KeyUpPressed = true;
            ControlEvent event(ControlEvent::Forward);
            ctl->dispatchEvent(event);
        }
        else if(!KeyDownPressed && key_event->keyval == GDK_KEY_Down)
        {
            KeyDownPressed = true;
            ControlEvent event(ControlEvent::Backward);
            ctl->dispatchEvent(event);
        }
        else if(!KeyLeftPressed && key_event->keyval == GDK_KEY_Left)
        {
            KeyLeftPressed = true;
            ControlEvent event(ControlEvent::RotateCCW);
            ctl->dispatchEvent(event);
        }
        else if(!KeyRightPressed && key_event->keyval == GDK_KEY_Right)
        {
            KeyRightPressed = true;
            ControlEvent event(ControlEvent::RotateCW);
            ctl->dispatchEvent(event);
        }
        else if(!spacePressed && key_event->keyval == GDK_KEY_space)
        {
            spacePressed = true;
            ControlEvent event(ControlEvent::Fire);
            ctl->dispatchEvent(event);
        }
        return Gtk::Window::on_key_press_event(key_event);
    }

    bool Window::on_key_release_event(GdkEventKey* key_event)
    {
        if(!ctl)
            return Gtk::Window::on_key_press_event(key_event);
        // Disable human control in RL training mode
        if(dynamic_cast<RLController*>(ctl.get()) != nullptr)
            return Gtk::Window::on_key_press_event(key_event);
        if(key_event->keyval == GDK_KEY_Up)
        {
            KeyUpPressed = false;
            ControlEvent event(ControlEvent::StopForward);
            ctl->dispatchEvent(event);
        }
        else if(key_event->keyval == GDK_KEY_Down)
        {
            KeyDownPressed = false;
            ControlEvent event(ControlEvent::StopBackward);
            ctl->dispatchEvent(event);
        }
        else if(key_event->keyval == GDK_KEY_Left)
        {
            KeyLeftPressed = false;
            ControlEvent event(ControlEvent::StopRotateCCW);
            ctl->dispatchEvent(event);
        }
        else if(key_event->keyval == GDK_KEY_Right)
        {
            KeyRightPressed = false;
            ControlEvent event(ControlEvent::StopRotateCW);
            ctl->dispatchEvent(event);
        }
        else if(key_event->keyval == GDK_KEY_space) {spacePressed = false;}
        return Gtk::Window::on_key_release_event(key_event);
    }

    Window::~Window() = default;
}
