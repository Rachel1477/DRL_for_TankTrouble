//
// RL Controller implementation for Agent vs SmithAI training
//

#include "RLController.h"
#include "Controller.h"
#include "LocalController.h"
#include "event/ControlEvent.h"
#include "Tank.h"
#include "Shell.h"
#include "util/Math.h"
#include <cmath>
#include <iostream>
#ifdef HAVE_PYBIND11
#include <pybind11/gil.h>
#endif

namespace TankTrouble
{
    static RLController *g_instance = nullptr;

    RLController::RLController(LocalController *shared)
        : ctl_shared_(shared ? shared : new LocalController()),
          own_controller_(!shared),
          episode_count_(0),
          total_reward_(0.0),
          agent_thread_(),
          episode_thread_()
    {
        g_instance = this;
    }

    RLController::~RLController()
    {
        if (g_instance == this)
            g_instance = nullptr;
        training_active_.store(false);
        if (agent_thread_.joinable())
            agent_thread_.join();
        if (episode_thread_.joinable())
            episode_thread_.join();
        if (own_controller_ && ctl_shared_)
        {
            delete ctl_shared_;
            ctl_shared_ = nullptr;
        }
    }

    RLController *RLController::getGlobalInstance()
    {
        return g_instance;
    }

    void RLController::start()
    {
        ctl_shared_->start();
        training_active_.store(true);
        if (agent_thread_.joinable())
            agent_thread_.join();
        if (episode_thread_.joinable())
            episode_thread_.join();
        agent_thread_ = std::thread(&RLController::agentLoop, this);
        episode_thread_ = std::thread(&RLController::episodeLoop, this);
        printf("[DEBUG] RLController::start() - agent and episode threads started\n");
    }

    void RLController::quitGame()
    {
        std::cout << "[RL] RLController::quitGame called, stopping threads..." << std::endl;
        training_active_.store(false);

        auto wait_with_timeout = [](std::thread &t, int timeout_ms)
        {
            if (!t.joinable())
                return;
            auto start = std::chrono::steady_clock::now();
            while (t.joinable())
            {
                if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(timeout_ms))
                {
                    std::cerr << "[RL] Thread join timeout, detaching..." << std::endl;
                    t.detach();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };

        if (agent_thread_.joinable())
        {
            std::cout << "[RL] Waiting for agent thread..." << std::endl;
            wait_with_timeout(agent_thread_, 2000);
        }
        if (episode_thread_.joinable())
        {
            std::cout << "[RL] Waiting for episode thread..." << std::endl;
            wait_with_timeout(episode_thread_, 2000);
        }

        std::cout << "[RL] All threads stopped" << std::endl;
        ctl_shared_->quitGame();
    }
    void RLController::dispatchEvent(const ControlEvent &event)
    {
        ctl_shared_->dispatchEvent(event);
    }
    void RLController::resetImmediate()
    {
        ctl_shared_->resetImmediate();
    }
    void RLController::stepOnce()
    {
        ctl_shared_->stepOnce();
    }
    int RLController::getSmithAction()
    {
        return ctl_shared_->getSmithAction();
    }
    int RLController::getAgentSmithAction()
    {
        return ctl_shared_->getRLSmithAction();
    }
    Controller::ObjectListPtr RLController::getObjects()
    {
        return ctl_shared_->getObjects();
    }
    Controller::BlockList *RLController::getBlocks()
    {
        return ctl_shared_->getBlocks();
    }
    std::vector<PlayerInfo> RLController::getPlaysInfo()
    {
        return ctl_shared_->getPlaysInfo();
    }
    void RLController::agentLoop()
    {
        using namespace std::chrono_literals;
        std::cout << "[RL] agentLoop entered, waiting for game initialization..." << std::endl;
        int wait_count = 0;
        while (training_active_.load() && wait_count < 100)
        {
            auto objsPtr = ctl_shared_->getObjects();
            if (objsPtr)
            {
                auto &objs = *objsPtr;
                if (objs.find(PLAYER_TANK_ID) != objs.end() && objs.find(AI_TANK_ID) != objs.end())
                {
                    std::cout << "[RL] agentLoop: Game initialized, starting agent loop" << std::endl;
                    break;
                }
            }
            std::this_thread::sleep_for(100ms);
            wait_count++;
        }
        if (wait_count >= 100)
        {
            std::cerr << "[RL] agentLoop: Timeout waiting for game initialization!" << std::endl;
            return;
        }
        auto last_log = std::chrono::steady_clock::now();
        int step_count = 0;
        std::vector<double> prev_state;
        int prev_action = 0;
        while (training_active_.load())
        {
            int action = 0;
            bool success = false;
            try
            {
                if (step_count == 0)
                    std::cout << "[RL] agentLoop: Building first state..." << std::endl;
                std::vector<double> s = buildState();
                if (step_count == 0)
                    std::cout << "[RL] agentLoop: First state built, size=" << s.size() << std::endl;
                if (s.empty() || s.size() != 82)
                {
                    std::cerr << "[RL] agentLoop: Invalid state size: " << s.size() << std::endl;
                    std::this_thread::sleep_for(50ms);
                    continue;
                }
                double dprog = 0, aprog = 0, spin = 0, stepc = 0;
                double r = dbgComputeReward(dprog, aprog, spin, stepc);
                if (step_count > 0 && step_cb_ && !prev_state.empty())
                {
                    try
                    {
                        bool done = false;
                        step_cb_(prev_state, prev_action, r, s, done);
                    }
                    catch (const std::exception &e)
                    {
                    }
                }
                if (get_action_cb_)
                {
                    if (step_count == 0)
                        std::cout << "[RL] agentLoop: Calling Python callback..." << std::endl;
                    try
                    {
                        action = get_action_cb_(s);
                        if (step_count == 0)
                            std::cout << "[RL] agentLoop: Python callback returned action=" << action << std::endl;
                        use_python_cb_ = true;
                        success = true;
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "[RL] agentLoop: Python callback exception: " << e.what() << std::endl;
                        action = decideActionFallback();
                        use_python_cb_ = false;
                    }
                }
                else
                {
                    if (step_count == 0)
                        std::cout << "[RL] agentLoop: Using fallback policy" << std::endl;
                    action = decideActionFallback();
                    use_python_cb_ = false;
                    success = true;
                }
                applyAgentAction(action);
                prev_state = s;
                prev_action = action;
                step_count++;
                if (step_count % 20 == 0 || step_count <= 5)
                {
                    std::cout << "[AGENT] step=" << step_count
                              << " action=" << action
                              << (use_python_cb_ ? " (python)" : " (fallback)")
                              << " r=" << r
                              << " dp=" << dprog << " ap=" << aprog
                              << " sp=" << spin << " sc=" << stepc << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[RL] agentLoop: Exception in main loop: " << e.what() << std::endl;
            }
            std::this_thread::sleep_for(50ms);
            if (std::chrono::steady_clock::now() - last_log > 5s)
            {
                last_log = std::chrono::steady_clock::now();
                std::cout << "[RL] agentLoop heartbeat: step=" << step_count
                          << ", python_cb=" << (use_python_cb_ ? "yes" : "no") << std::endl;
            }
        }
        std::cout << "[RL] agentLoop exit after " << step_count << " steps" << std::endl;
    }

    int RLController::decideActionFallback()
    {
        auto objsPtr = ctl_shared_->getObjects();
        auto &objs = *objsPtr;
        if (objs.find(PLAYER_TANK_ID) == objs.end() || objs.find(AI_TANK_ID) == objs.end())
            return 0;
        auto *me = dynamic_cast<Tank *>(objs[PLAYER_TANK_ID].get());
        auto *enemy = dynamic_cast<Tank *>(objs[AI_TANK_ID].get());
        auto my = me->getCurrentPosition();
        auto en = enemy->getCurrentPosition();
        double dx = en.pos.x() - my.pos.x();
        double dy = en.pos.y() - my.pos.y();
        double toEnemyAngle = std::atan2(dy, dx) * 180.0 / M_PI;
        auto norm = [](double a)
        { while(a < 0) a += 360.0; while(a >= 360.0) a -= 360.0; return a; };
        double myAng = norm(my.angle);
        double tgtAng = norm(toEnemyAngle);
        double diff = norm(tgtAng - myAng);
        if (std::fabs(diff) < 10.0 || std::fabs(360.0 - diff) < 10.0)
        {
            if (me->remainShells() > 0 && (std::hypot(dx, dy) < 250.0))
                return 5;
            return 1;
        }
        if (diff > 180.0)
            return 4;
        return 3;
    }

    void RLController::applyAgentAction(int action)
    {
        switch (action)
        {
        case 0:
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::StopForward));
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::StopBackward));
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::StopRotateCW));
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::StopRotateCCW));
            break;
        case 1:
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::Forward));
            break;
        case 2:
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::Backward));
            break;
        case 3:
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::RotateCW));
            break;
        case 4:
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::RotateCCW));
            break;
        case 5:
            ctl_shared_->dispatchEvent(ControlEvent(ControlEvent::Fire));
            break;
        default:
            break;
        }
    }

    void RLController::episodeLoop()
    {
        using namespace std::chrono_literals;
        bool prev_agent_alive = true, prev_smith_alive = true;
        while (training_active_.load())
        {
            auto objsPtr = ctl_shared_->getObjects();
            auto &objs = *objsPtr;
            bool agent_alive = objs.find(PLAYER_TANK_ID) != objs.end();
            bool smith_alive = objs.find(AI_TANK_ID) != objs.end();
            if ((!agent_alive || !smith_alive) && (prev_agent_alive && prev_smith_alive))
            {
                episode_count_++;
                double total_reward = total_reward_;
                bool agent_won = agent_alive && !smith_alive;
                std::cout << "[RL] Episode " << episode_count_ << " finished. Score: " << total_reward << (agent_won ? " (WIN)" : " (LOSE)") << std::endl;
                if (episode_end_cb_)
                {
                    try
                    {
                        episode_end_cb_(episode_count_, total_reward, agent_won);
                    }
                    catch (...)
                    {
                    }
                }
                total_reward_ = 0.0;
            }
            prev_agent_alive = agent_alive;
            prev_smith_alive = smith_alive;
            std::this_thread::sleep_for(100ms);
        }
    }

    std::vector<double> RLController::buildState()
    {
        std::vector<double> state;
        auto objsPtr = ctl_shared_->getObjects();
        auto &objs = *objsPtr;
        if (objs.find(PLAYER_TANK_ID) == objs.end() || objs.find(AI_TANK_ID) == objs.end())
            return std::vector<double>(9 + 1 + 72, 0.0);
        auto *me = dynamic_cast<Tank *>(objs[PLAYER_TANK_ID].get());
        auto *enemy = dynamic_cast<Tank *>(objs[AI_TANK_ID].get());
        auto my = me->getCurrentPosition();
        auto en = enemy->getCurrentPosition();
        auto normx = [](double x)
        { return x / static_cast<double>(GAME_VIEW_WIDTH); };
        auto normy = [](double y)
        { return y / static_cast<double>(GAME_VIEW_HEIGHT); };
        state.push_back(normx(my.pos.x()));
        state.push_back(normy(my.pos.y()));
        double s, c;
        s = std::sin(my.angle * M_PI / 180.0);
        c = std::cos(my.angle * M_PI / 180.0);
        state.push_back(s);
        state.push_back(c);
        state.push_back(me->remainShells() > 0 ? 1.0 : 0.0);
        state.push_back(normx(en.pos.x() - my.pos.x()));
        state.push_back(normy(en.pos.y() - my.pos.y()));
        double se = std::sin(en.angle * M_PI / 180.0), ce = std::cos(en.angle * M_PI / 180.0);
        state.push_back(se);
        state.push_back(ce);
        state.push_back(0.0);
        const int NUM_RAYS = 72;
        const double MAX_DIST = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        for (int i = 0; i < NUM_RAYS; i++)
        {
            double ang = (360.0 / NUM_RAYS) * i;
            double rad = ang * M_PI / 180.0;
            double dx = std::cos(rad), dy = std::sin(rad);
            double step = 4.0;
            double wallD = MAX_DIST;
            for (double t = 0.0; t <= MAX_DIST; t += step)
            {
                util::Vec p(my.pos.x() + dx * t, my.pos.y() + dy * t);
                if (p.x() <= 0 || p.x() >= GAME_VIEW_WIDTH || p.y() <= 0 || p.y() >= GAME_VIEW_HEIGHT)
                {
                    wallD = t;
                    break;
                }
                for (const auto &kv : *ctl_shared_->getBlocks())
                {
                    const Block &b = kv.second;
                    auto bc = b.center();
                    if (std::abs(p.x() - bc.x()) < b.width() / 2 && std::abs(p.y() - bc.y()) < b.height() / 2)
                    {
                        wallD = t;
                        break;
                    }
                }
            }
            state.push_back(wallD / MAX_DIST);
        }
        return state;
    }

    void RLController::dbgInitMemory()
    {
        auto objsPtr = ctl_shared_->getObjects();
        auto &objs = *objsPtr;
        if (objs.find(PLAYER_TANK_ID) == objs.end() || objs.find(AI_TANK_ID) == objs.end())
            return;
        auto *me = dynamic_cast<Tank *>(objs[PLAYER_TANK_ID].get());
        auto *enemy = dynamic_cast<Tank *>(objs[AI_TANK_ID].get());
        auto my = me->getCurrentPosition();
        auto en = enemy->getCurrentPosition();
        dbg_last_my_pos_ = my.pos;
        dbg_last_my_angle_ = my.angle;
        dbg_last_enemy_pos_ = en.pos;
        double maxd = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        double dist = std::hypot(en.pos.x() - my.pos.x(), en.pos.y() - my.pos.y());
        dbg_prev_dist_norm_ = dist / maxd;
        double bearing = std::atan2(en.pos.y() - my.pos.y(), en.pos.x() - my.pos.x()) * 180.0 / M_PI;
        double diff = std::fmod(std::fabs(bearing - my.angle), 360.0);
        if (diff > 180.0)
            diff = 360.0 - diff;
        dbg_prev_align_norm_ = diff / 180.0;
        dbg_initialized_ = true;
    }

    double RLController::dbgComputeReward(double &dist_prog, double &align_prog, double &spin_pen, double &step_cost)
    {
        if (!dbg_initialized_)
            dbgInitMemory();
        auto objsPtr = ctl_shared_->getObjects();
        auto &objs = *objsPtr;
        if (objs.find(PLAYER_TANK_ID) == objs.end() || objs.find(AI_TANK_ID) == objs.end())
            return 0.0;
        auto *me = dynamic_cast<Tank *>(objs[PLAYER_TANK_ID].get());
        auto *enemy = dynamic_cast<Tank *>(objs[AI_TANK_ID].get());
        auto my = me->getCurrentPosition();
        auto en = enemy->getCurrentPosition();
        double maxd = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        double dist = std::hypot(en.pos.x() - my.pos.x(), en.pos.y() - my.pos.y());
        double dist_norm = dist / maxd;
        double bearing = std::atan2(en.pos.y() - my.pos.y(), en.pos.x() - my.pos.x()) * 180.0 / M_PI;
        double diff = std::fmod(std::fabs(bearing - my.angle), 360.0);
        if (diff > 180.0)
            diff = 360.0 - diff;
        double align_norm = diff / 180.0;
        dist_prog = 0.5 * (dbg_prev_dist_norm_ - dist_norm);
        align_prog = 0.1 * (dbg_prev_align_norm_ - align_norm);
        double move_dist = std::hypot(my.pos.x() - dbg_last_my_pos_.x(), my.pos.y() - dbg_last_my_pos_.y());
        double angle_change = std::fmod(std::fabs(my.angle - dbg_last_my_angle_), 360.0);
        if (angle_change > 180.0)
            angle_change = 360.0 - angle_change;
        spin_pen = (move_dist < 1.0 && angle_change > 5.0) ? -0.02 : 0.0;
        step_cost = -0.001;
        double r = dist_prog + align_prog + spin_pen + step_cost;
        dbg_last_my_pos_ = my.pos;
        dbg_last_my_angle_ = my.angle;
        dbg_last_enemy_pos_ = en.pos;
        dbg_prev_dist_norm_ = dist_norm;
        dbg_prev_align_norm_ = align_norm;
        return r;
    }
}
