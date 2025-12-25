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
    // Python对象作为文件作用域static变量，以便在lambda中访问
#ifdef HAVE_PYBIND11
    namespace py = pybind11;
    static py::object g_sys_module;
    static py::object g_trainer_module;
    static py::object g_tank_env_module;
    static py::object g_py_env;  // 保存TankEnv实例，避免在GIL释放后被销毁
    static py::object g_get_action_py;
    static py::object g_on_episode_end_py;
    static py::object g_on_step_py;
    static bool g_python_initialized = false;
    static PyThreadState* g_main_thread_state = nullptr;  // 主线程状态
#endif
    Window::Window() : localCtl(nullptr), ctl(nullptr), rlCtl(nullptr),
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

    void Window::notifyLoginSuccess() { loginSuccessNotifier.emit(); }

    void Window::notifyRoomUpdate() { roomUpdateNotifier.emit(); }

    void Window::notifyGameOn() { gameOnNotifier.emit(); }

    void Window::notifyGameOff() { gameOffNotifier.emit(); }

    void Window::onUserChooseLocal()
    {
        remove();
        if (!localCtl)
            localCtl = std::make_unique<LocalController>();
        ctl = localCtl.get();  // 只是观察指针，不拥有所有权
        ctl->start();
        gameView = std::make_unique<GameView>(ctl);
        gameView->signal_quit_game().connect(sigc::mem_fun(*this, &Window::toEntryView));
        add(*gameView);
        gameView->show();
    }

    void Window::onUserChooseRLTraining()
    {
        remove();
        if (!localCtl)
            localCtl = std::make_unique<LocalController>();
        if (!rlCtl)
            rlCtl = std::make_unique<RLController>(localCtl.get());
        ctl = rlCtl.get();  // 只是观察指针，不拥有所有权

        // Inject Python DQN callbacks
#ifdef HAVE_PYBIND11
        static std::unique_ptr<pybind11::scoped_interpreter> guard;
        if (!guard)
        {
            // 设置 CONDA 环境变量，确保嵌入式 Python 使用 conda 环境
            setenv("PYTHONHOME", CONDA_PY_PREFIX, 1);
            std::string site = std::string(CONDA_PY_PREFIX) + "/lib/python" + CONDA_PY_VER + "/site-packages";
            // 追加工程模块路径与 conda site-packages 到 PYTHONPATH
            const char *oldpp = getenv("PYTHONPATH");
            std::string newpp = site + ":" + std::string(PROJECT_BUILD_DIR) + ":" + std::string(PROJECT_ROOT_DIR) + "/build:" + std::string(PROJECT_ROOT_DIR) + "/cmake-build-debug:" + std::string(PROJECT_ROOT_DIR) + "/python";
            if (oldpp && *oldpp)
                newpp = newpp + ":" + std::string(oldpp);
            setenv("PYTHONPATH", newpp.c_str(), 1);
            guard = std::make_unique<pybind11::scoped_interpreter>();
        }
        
        try
        {
            namespace py = pybind11;

            // 如果是第二次及以后进入，需要先获取GIL才能清理和重新初始化Python对象
            std::unique_ptr<py::gil_scoped_acquire> gil_guard;
            if (g_python_initialized)
            {
                std::cout << "[RL] Cleaning up previous Python environment..." << std::endl;
                std::cout << "[RL] Acquiring GIL for cleanup and re-initialization..." << std::endl;
                gil_guard = std::make_unique<py::gil_scoped_acquire>();
                
                // 释放旧的Python对象引用（现在GIL已持有）
                g_get_action_py = py::object();
                g_on_episode_end_py = py::object();
                g_on_step_py = py::object();
                g_py_env = py::object();  // 释放TankEnv实例
                g_trainer_module = py::object();
                g_tank_env_module = py::object();
                // g_sys_module可以保留，因为它不持有C++对象引用
                
                std::cout << "[RL] Python objects cleaned up" << std::endl;
            }
            // 注意：不要在这里释放gil_guard，我们需要保持GIL直到完成所有Python初始化

            if (!g_sys_module)
            {
                g_sys_module = py::module_::import("sys");
                py::list path = g_sys_module.attr("path");
                // Add absolute project paths to sys.path to avoid WD issues
                path.append(PROJECT_BUILD_DIR);
                path.append(PROJECT_ROOT_DIR "/build");
                path.append(PROJECT_ROOT_DIR "/cmake-build-debug");
                path.append(PROJECT_ROOT_DIR "/python");
            }

            // 重新创建Python环境引用
            std::cout << "[RL] Creating new Python environment..." << std::endl;
            g_tank_env_module = py::module_::import("tank_trouble_env");
            g_py_env = g_tank_env_module.attr("TankEnv")(py::cast(rlCtl.get(), py::return_value_policy::reference));
            g_trainer_module = py::module_::import("train_with_gui");
            g_trainer_module.attr("set_global_env")(g_py_env);
            
            // initialize agent (state=82, action=6)
            std::cout << "[RL] Initializing agent..." << std::endl;
            try
            {
                g_trainer_module.attr("initialize_agent")(82, 6);
            }
            catch (const std::exception &e)
            {
                std::cerr << "[RL] initialize_agent exception: " << e.what() << std::endl;
            }

            // Store callbacks
            std::cout << "[RL] Getting Python callbacks..." << std::endl;
            g_get_action_py = g_trainer_module.attr("get_action_from_state");
            g_on_episode_end_py = g_trainer_module.attr("on_episode_end");
            g_on_step_py = g_trainer_module.attr("on_step");

            g_python_initialized = true;

            // Wrap in lambdas that acquire GIL
            // 使用文件作用域的全局变量，每次重新进入时会被更新
            std::cout << "[RL] Creating callback wrappers..." << std::endl;
            auto get_action_cb = [](const std::vector<double> &state) -> int
            {
                py::gil_scoped_acquire acquire;
                return g_get_action_py(state).cast<int>();
            };

            auto episode_end_cb = [](int episode, double reward, bool won)
            {
                py::gil_scoped_acquire acquire;
                g_on_episode_end_py(episode, reward, won);
            };

            auto step_cb = [](const std::vector<double> &prev_state, int prev_action,
                              double reward, const std::vector<double> &next_state, bool done)
            {
                py::gil_scoped_acquire acquire;
                g_on_step_py(prev_state, prev_action, reward, next_state, done);
            };

            static_cast<RLController *>(ctl)->setGetActionCallback(get_action_cb);
            static_cast<RLController *>(ctl)->setEpisodeEndCallback(episode_end_cb);
            static_cast<RLController *>(ctl)->setStepCallback(step_cb);
            std::cout << "[RL] Python callbacks injected successfully" << std::endl;
            
            // 注意：不要在这里释放gil_guard
            // 它会在try块结束时自动析构，释放GIL（如果是第二次及以后进入）
            // 第一次进入时gil_guard未创建，GIL由scoped_interpreter持有
        }
        catch (const std::exception &e)
        {
            std::cerr << "[RL] Python callback injection failed: " << e.what() << std::endl;
        }

        // CRITICAL: Release GIL to allow agentLoop thread to acquire it
        // 在try块结束后，GIL的状态：
        // - 第一次进入：由scoped_interpreter持有
        // - 第二次进入：gil_guard已析构并释放GIL
        // 所以第一次需要释放，第二次已经释放了
        std::cout << "[RL] Releasing GIL for worker threads..." << std::endl;
        if (!g_main_thread_state)
        {
            // 第一次进入，需要释放GIL
            g_main_thread_state = PyEval_SaveThread();
            std::cout << "[RL] GIL released (first time)" << std::endl;
        }
        else
        {
            // 第二次及以后进入，GIL已被gil_guard释放，无需再释放
            std::cout << "[RL] GIL already released by gil_guard" << std::endl;
        }
#endif

        // Start controller and show view
        std::cout << "[RL] Starting controller..." << std::endl;
        ctl->start();
        std::cout << "[RL] Controller started, creating game view..." << std::endl;
        gameView = std::make_unique<GameView>(ctl);
        std::cout << "[RL] Game view created, connecting signals..." << std::endl;
        gameView->signal_quit_game().connect(sigc::mem_fun(*this, &Window::toEntryView));
        std::cout << "[RL] Adding game view to window..." << std::endl;
        add(*gameView);
        std::cout << "[RL] Showing game view..." << std::endl;
        gameView->show();
        std::cout << "[RL] onUserChooseRLTraining completed" << std::endl;
    }

    void Window::toEntryView()
    {
        remove();
        if (gameView)
            gameView.reset();
        
        // 显式停止并销毁controllers（会停止所有线程）
        ctl = nullptr;
        if (rlCtl)
            rlCtl.reset();
        if (localCtl)
            localCtl.reset();
        
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
        gameView = std::make_unique<GameView>(ctl);
        add(*gameView);
        gameView->signal_quit_game().connect(sigc::mem_fun(*this, &Window::onGameOff));
        gameView->show();
    }

    void Window::onGameOff()
    {
        // on quitting local game, return to entry view
        toEntryView();
    }

    bool Window::on_key_press_event(GdkEventKey *key_event)
    {
        if (!ctl)
            return Gtk::Window::on_key_press_event(key_event);
        // Disable human control in RL training mode
        if (dynamic_cast<RLController *>(ctl) != nullptr)
            return Gtk::Window::on_key_press_event(key_event);
        if (!KeyUpPressed && key_event->keyval == GDK_KEY_Up)
        {
            KeyUpPressed = true;
            ControlEvent event(ControlEvent::Forward);
            ctl->dispatchEvent(event);
        }
        else if (!KeyDownPressed && key_event->keyval == GDK_KEY_Down)
        {
            KeyDownPressed = true;
            ControlEvent event(ControlEvent::Backward);
            ctl->dispatchEvent(event);
        }
        else if (!KeyLeftPressed && key_event->keyval == GDK_KEY_Left)
        {
            KeyLeftPressed = true;
            ControlEvent event(ControlEvent::RotateCCW);
            ctl->dispatchEvent(event);
        }
        else if (!KeyRightPressed && key_event->keyval == GDK_KEY_Right)
        {
            KeyRightPressed = true;
            ControlEvent event(ControlEvent::RotateCW);
            ctl->dispatchEvent(event);
        }
        else if (!spacePressed && key_event->keyval == GDK_KEY_space)
        {
            spacePressed = true;
            ControlEvent event(ControlEvent::Fire);
            ctl->dispatchEvent(event);
        }
        return Gtk::Window::on_key_press_event(key_event);
    }

    bool Window::on_key_release_event(GdkEventKey *key_event)
    {
        if (!ctl)
            return Gtk::Window::on_key_press_event(key_event);
        // Disable human control in RL training mode
        if (dynamic_cast<RLController *>(ctl) != nullptr)
            return Gtk::Window::on_key_press_event(key_event);
        if (key_event->keyval == GDK_KEY_Up)
        {
            KeyUpPressed = false;
            ControlEvent event(ControlEvent::StopForward);
            ctl->dispatchEvent(event);
        }
        else if (key_event->keyval == GDK_KEY_Down)
        {
            KeyDownPressed = false;
            ControlEvent event(ControlEvent::StopBackward);
            ctl->dispatchEvent(event);
        }
        else if (key_event->keyval == GDK_KEY_Left)
        {
            KeyLeftPressed = false;
            ControlEvent event(ControlEvent::StopRotateCCW);
            ctl->dispatchEvent(event);
        }
        else if (key_event->keyval == GDK_KEY_Right)
        {
            KeyRightPressed = false;
            ControlEvent event(ControlEvent::StopRotateCW);
            ctl->dispatchEvent(event);
        }
        else if (key_event->keyval == GDK_KEY_space)
        {
            spacePressed = false;
        }
        return Gtk::Window::on_key_release_event(key_event);
    }

    Window::~Window() = default;
}
