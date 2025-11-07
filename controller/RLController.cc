//
// RL Controller implementation for Agent vs SmithAI training
//

#include "RLController.h"
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
    RLController::RLController():
        LocalController(),
        episode_count_(0),
        total_reward_(0.0)
    {}

    RLController::~RLController()
    {
        training_active_.store(false);
    }

    void RLController::start()
    {
        training_active_.store(true);
        LocalController::start();
        // Launch agent decision loop
        std::cout << "[RL] RLController::start -> launching agent and episode threads" << std::endl;
        agent_thread_ = std::thread(&RLController::agentLoop, this);
        episode_thread_ = std::thread(&RLController::episodeLoop, this);
    }

    void RLController::quitGame()
    {
        std::cout << "[RL] RLController::quitGame called, stopping threads..." << std::endl;
        training_active_.store(false);
        
        // Wait for threads to finish with timeout
        auto wait_with_timeout = [](std::thread& t, int timeout_ms) {
            if(!t.joinable()) return;
            auto start = std::chrono::steady_clock::now();
            while(t.joinable()) {
                if(std::chrono::steady_clock::now() - start > std::chrono::milliseconds(timeout_ms)) {
                    std::cerr << "[RL] Thread join timeout, detaching..." << std::endl;
                    t.detach();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };
        
        if(agent_thread_.joinable()) {
            std::cout << "[RL] Waiting for agent thread..." << std::endl;
            agent_thread_.join();
        }
        if(episode_thread_.joinable()) {
            std::cout << "[RL] Waiting for episode thread..." << std::endl;
            episode_thread_.join();
        }
        
        std::cout << "[RL] All threads stopped" << std::endl;
        LocalController::quitGame();
    }
    // No extra logic here to avoid accessing LocalController's private members.

    // Agent loop: periodically decide an action and dispatch events
    void RLController::agentLoop()
    {
        using namespace std::chrono_literals;
        std::cout << "[RL] agentLoop entered, waiting for game initialization..." << std::endl;
        
        // Wait for game to be ready (objects initialized)
        int wait_count = 0;
        while(training_active_.load() && wait_count < 100)
        {
            auto objsPtr = getObjects();
            if(objsPtr)
            {
                auto& objs = *objsPtr;
                if(objs.find(PLAYER_TANK_ID) != objs.end() && objs.find(AI_TANK_ID) != objs.end())
                {
                    std::cout << "[RL] agentLoop: Game initialized, starting agent loop" << std::endl;
                    break;
                }
            }
            std::this_thread::sleep_for(100ms);
            wait_count++;
        }
        
        if(wait_count >= 100)
        {
            std::cerr << "[RL] agentLoop: Timeout waiting for game initialization!" << std::endl;
            return;
        }
        
        auto last_log = std::chrono::steady_clock::now();
        int step_count = 0;
        
        std::vector<double> prev_state;
        int prev_action = 0;
        
        while(training_active_.load())
        {
            int action = 0;
            bool success = false;
            
            try {
                if(step_count == 0) std::cout << "[RL] agentLoop: Building first state..." << std::endl;
                std::vector<double> s = buildState();
                if(step_count == 0) std::cout << "[RL] agentLoop: First state built, size=" << s.size() << std::endl;
                if(s.empty() || s.size() != 122)  // Updated state size
                {
                    std::cerr << "[RL] agentLoop: Invalid state size: " << s.size() << std::endl;
                    std::this_thread::sleep_for(50ms);
                    continue;
                }
                
                // Calculate reward from previous step (if exists)
                double dprog=0, aprog=0, spin=0, stepc=0;
                double r = dbgComputeReward(dprog, aprog, spin, stepc);
                
                // If we have a previous state, send the experience to Python
                if(step_count > 0 && step_cb_ && !prev_state.empty())
                {
                    try {
                        bool done = false;  // Will be detected by episode_end callback
                        step_cb_(prev_state, prev_action, r, s, done);
                    } catch(const std::exception& e) {
                        // Silently ignore step callback errors
                    }
                }
                
                // If Python callback provided, use it
                if(get_action_cb_)
                {
                    if(step_count == 0) std::cout << "[RL] agentLoop: Calling Python callback..." << std::endl;
                    try {
                        // GIL acquisition is handled inside the lambda wrapper
                        action = get_action_cb_(s);
                        if(step_count == 0) std::cout << "[RL] agentLoop: Python callback returned action=" << action << std::endl;
                        use_python_cb_ = true;
                        success = true;
                    } catch(const std::exception& e) {
                        std::cerr << "[RL] agentLoop: Python callback exception: " << e.what() << std::endl;
                        action = decideActionFallback();
                        use_python_cb_ = false;
                    }
                }
                else
                {
                    if(step_count == 0) std::cout << "[RL] agentLoop: Using fallback policy" << std::endl;
                    action = decideActionFallback();
                    use_python_cb_ = false;
                    success = true;
                }

                applyAgentAction(action);
                
                // Store for next iteration
                prev_state = s;
                prev_action = action;
                
                step_count++;
                if(step_count % 20 == 0 || step_count <= 5)  // Log first 5 steps and every 20th step
                {
                    std::cout << "[AGENT] step=" << step_count
                              << " action=" << action
                              << (use_python_cb_ ? " (python)" : " (fallback)")
                              << " r=" << r
                              << " dp=" << dprog << " ap=" << aprog
                              << " sp=" << spin << " sc=" << stepc << std::endl;
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << "[RL] agentLoop: Exception in main loop: " << e.what() << std::endl;
            }
            
            std::this_thread::sleep_for(50ms);

            if(std::chrono::steady_clock::now() - last_log > 5s)
            {
                last_log = std::chrono::steady_clock::now();
                std::cout << "[RL] agentLoop heartbeat: step=" << step_count
                          << ", python_cb=" << (use_python_cb_ ? "yes" : "no") << std::endl;
            }
        }
        std::cout << "[RL] agentLoop exit after " << step_count << " steps" << std::endl;
    }

    // Fallback heuristic policy aiming at SmithAI tank
    int RLController::decideActionFallback()
    {
        auto objsPtr = getObjects();
        auto& objs = *objsPtr;
        if(objs.find(PLAYER_TANK_ID) == objs.end() || objs.find(AI_TANK_ID) == objs.end())
            return 0; // do nothing if missing
        auto* me = dynamic_cast<Tank*>(objs[PLAYER_TANK_ID].get());
        auto* enemy = dynamic_cast<Tank*>(objs[AI_TANK_ID].get());
        auto my = me->getCurrentPosition();
        auto en = enemy->getCurrentPosition();

        double dx = en.pos.x() - my.pos.x();
        double dy = en.pos.y() - my.pos.y();
        double toEnemyAngle = std::atan2(dy, dx) * 180.0 / M_PI;
        // Normalize angles to [0,360)
        auto norm = [](double a){ while(a < 0) a += 360.0; while(a >= 360.0) a -= 360.0; return a; };
        double myAng = norm(my.angle);
        double tgtAng = norm(toEnemyAngle);
        double diff = norm(tgtAng - myAng);

        // If roughly aligned, move forward and shoot
        if(std::fabs(diff) < 10.0 || std::fabs(360.0 - diff) < 10.0)
        {
            if(me->remainShells() > 0 && (std::hypot(dx, dy) < 250.0))
                return 5; // SHOOT
            return 1; // MOVE_FORWARD
        }
        // Rotate towards target: choose CW or CCW to minimize angle
        if(diff > 180.0) return 4; // ROTATE_CCW
        return 3; // ROTATE_CW
    }

    void RLController::applyAgentAction(int action)
    {
        switch(action)
        {
            case 0: // DO_NOTHING
                dispatchEvent(ControlEvent(ControlEvent::StopForward));
                dispatchEvent(ControlEvent(ControlEvent::StopBackward));
                dispatchEvent(ControlEvent(ControlEvent::StopRotateCW));
                dispatchEvent(ControlEvent(ControlEvent::StopRotateCCW));
                break;
            case 1: // MOVE_FORWARD
                dispatchEvent(ControlEvent(ControlEvent::Forward));
                break;
            case 2: // MOVE_BACKWARD
                dispatchEvent(ControlEvent(ControlEvent::Backward));
                break;
            case 3: // ROTATE_CW
                dispatchEvent(ControlEvent(ControlEvent::RotateCW));
                break;
            case 4: // ROTATE_CCW
                dispatchEvent(ControlEvent(ControlEvent::RotateCCW));
                break;
            case 5: // SHOOT
                dispatchEvent(ControlEvent(ControlEvent::Fire));
                break;
            default:
                break;
        }
    }

    // Episode monitoring: detect death and notify callback
    void RLController::episodeLoop()
    {
        using namespace std::chrono_literals;
        bool prev_agent_alive = true, prev_smith_alive = true;
        while(training_active_.load())
        {
            auto objsPtr = getObjects();
            auto& objs = *objsPtr;
            bool agent_alive = objs.find(PLAYER_TANK_ID) != objs.end();
            bool smith_alive = objs.find(AI_TANK_ID) != objs.end();

            // Episode end when either died and previously both alive
            if((!agent_alive || !smith_alive) && (prev_agent_alive && prev_smith_alive))
            {
                episode_count_++;
                double total_reward = 0.0; // reward can be computed in Python if needed
                bool agent_won = agent_alive && !smith_alive;
                if(episode_end_cb_) {
                    try { episode_end_cb_(episode_count_, total_reward, agent_won); } catch(...) {}
                }
            }

            prev_agent_alive = agent_alive;
            prev_smith_alive = smith_alive;
            std::this_thread::sleep_for(100ms);
        }
    }

    // Build 122-dim state like TankEnv (updated with map grid)
    std::vector<double> RLController::buildState()
    {
        std::vector<double> state;
        auto objsPtr = getObjects();
        auto& objs = *objsPtr;
        if(objs.find(PLAYER_TANK_ID) == objs.end() || objs.find(AI_TANK_ID) == objs.end())
            return std::vector<double>(122, 0.0);  // Updated dimension

        auto* me = dynamic_cast<Tank*>(objs[PLAYER_TANK_ID].get());
        auto* enemy = dynamic_cast<Tank*>(objs[AI_TANK_ID].get());
        auto my = me->getCurrentPosition();
        auto en = enemy->getCurrentPosition();

        auto normx = [](double x){ return x / static_cast<double>(GAME_VIEW_WIDTH); };
        auto normy = [](double y){ return y / static_cast<double>(GAME_VIEW_HEIGHT); };

        state.push_back(normx(my.pos.x()));
        state.push_back(normy(my.pos.y()));
        double s, c; s = std::sin(my.angle * M_PI / 180.0); c = std::cos(my.angle * M_PI / 180.0);
        state.push_back(s); state.push_back(c);
        state.push_back(me->remainShells() > 0 ? 1.0 : 0.0);

        state.push_back(normx(en.pos.x() - my.pos.x()));
        state.push_back(normy(en.pos.y() - my.pos.y()));
        double se = std::sin(en.angle * M_PI / 180.0), ce = std::cos(en.angle * M_PI / 180.0);
        state.push_back(se); state.push_back(ce);

        // Global map grid (8x8 = 64 cells)
        const int MAP_GRID_SIZE = 8;
        const double CELL_WIDTH = GAME_VIEW_WIDTH / MAP_GRID_SIZE;
        const double CELL_HEIGHT = GAME_VIEW_HEIGHT / MAP_GRID_SIZE;
        auto* blocks = getBlocks();
        std::vector<double> map_grid(MAP_GRID_SIZE * MAP_GRID_SIZE, 0.0);
        for (const auto& kv : *blocks)
        {
            const Block& b = kv.second;
            auto bc = b.center();
            double bw = b.width();
            double bh = b.height();
            int min_gx = std::max(0, static_cast<int>((bc.x() - bw/2) / CELL_WIDTH));
            int max_gx = std::min(static_cast<int>(MAP_GRID_SIZE-1), static_cast<int>((bc.x() + bw/2) / CELL_WIDTH));
            int min_gy = std::max(0, static_cast<int>((bc.y() - bh/2) / CELL_HEIGHT));
            int max_gy = std::min(static_cast<int>(MAP_GRID_SIZE-1), static_cast<int>((bc.y() + bh/2) / CELL_HEIGHT));
            for (int gy = min_gy; gy <= max_gy; gy++)
            {
                for (int gx = min_gx; gx <= max_gx; gx++)
                {
                    map_grid[gy * MAP_GRID_SIZE + gx] = 1.0;
                }
            }
        }
        state.insert(state.end(), map_grid.begin(), map_grid.end());
        
        // Line-of-sight flag (placeholder: always 0 for now in RLController)
        state.push_back(0.0);  // Can implement hasDirectLineToEnemy if needed

        // Rays
        const int NUM_RAYS = 16; const double MAX_DIST = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        for(int i = 0; i < NUM_RAYS; i++)
        {
            double ang = (360.0 / NUM_RAYS) * i; double rad = ang * M_PI / 180.0;
            double dx = std::cos(rad), dy = std::sin(rad);
            double step = 4.0; double wallD = MAX_DIST, enemyD = MAX_DIST, bulletD = MAX_DIST;
            for(double t = 0.0; t <= MAX_DIST; t += step)
            {
                util::Vec p(my.pos.x() + dx * t, my.pos.y() + dy * t);
                if(p.x() <= 0 || p.x() >= GAME_VIEW_WIDTH || p.y() <= 0 || p.y() >= GAME_VIEW_HEIGHT)
                { wallD = t; break; }
                for(const auto& kv : *blocks)
                {
                    const Block& b = kv.second; auto bc = b.center();
                    if(std::abs(p.x() - bc.x()) < b.width()/2 && std::abs(p.y() - bc.y()) < b.height()/2)
                    { wallD = t; break; }
                }
                if(std::abs(p.x() - en.pos.x()) < Tank::TANK_WIDTH/2 && std::abs(p.y() - en.pos.y()) < Tank::TANK_HEIGHT/2)
                { enemyD = t; }
            }
            state.push_back(wallD / MAX_DIST);
            state.push_back(enemyD < MAX_DIST ? enemyD / MAX_DIST : 1.0);
            state.push_back(bulletD < MAX_DIST ? bulletD / MAX_DIST : 1.0);
        }
        return state;
    }

    void RLController::dbgInitMemory()
    {
        auto objsPtr = getObjects();
        auto& objs = *objsPtr;
        if(objs.find(PLAYER_TANK_ID) == objs.end() || objs.find(AI_TANK_ID) == objs.end()) return;
        auto* me = dynamic_cast<Tank*>(objs[PLAYER_TANK_ID].get());
        auto* enemy = dynamic_cast<Tank*>(objs[AI_TANK_ID].get());
        auto my = me->getCurrentPosition(); auto en = enemy->getCurrentPosition();
        dbg_last_my_pos_ = my.pos; dbg_last_my_angle_ = my.angle; dbg_last_enemy_pos_ = en.pos;
        double maxd = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        double dist = std::hypot(en.pos.x() - my.pos.x(), en.pos.y() - my.pos.y());
        dbg_prev_dist_norm_ = dist / maxd;
        double bearing = std::atan2(en.pos.y() - my.pos.y(), en.pos.x() - my.pos.x()) * 180.0 / M_PI;
        double diff = std::fmod(std::fabs(bearing - my.angle), 360.0); if(diff > 180.0) diff = 360.0 - diff;
        dbg_prev_align_norm_ = diff / 180.0;
        dbg_initialized_ = true;
    }

    double RLController::dbgComputeReward(double& dist_prog, double& align_prog, double& spin_pen, double& step_cost)
    {
        if(!dbg_initialized_) dbgInitMemory();
        auto objsPtr = getObjects(); auto& objs = *objsPtr;
        if(objs.find(PLAYER_TANK_ID) == objs.end() || objs.find(AI_TANK_ID) == objs.end()) return 0.0;
        auto* me = dynamic_cast<Tank*>(objs[PLAYER_TANK_ID].get());
        auto* enemy = dynamic_cast<Tank*>(objs[AI_TANK_ID].get());
        auto my = me->getCurrentPosition(); auto en = enemy->getCurrentPosition();
        double maxd = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        double dist = std::hypot(en.pos.x() - my.pos.x(), en.pos.y() - my.pos.y());
        double dist_norm = dist / maxd;
        double bearing = std::atan2(en.pos.y() - my.pos.y(), en.pos.x() - my.pos.x()) * 180.0 / M_PI;
        double diff = std::fmod(std::fabs(bearing - my.angle), 360.0); if(diff > 180.0) diff = 360.0 - diff;
        double align_norm = diff / 180.0;

        dist_prog = 0.5 * (dbg_prev_dist_norm_ - dist_norm);
        align_prog = 0.1 * (dbg_prev_align_norm_ - align_norm);

        double move_dist = std::hypot(my.pos.x() - dbg_last_my_pos_.x(), my.pos.y() - dbg_last_my_pos_.y());
        double angle_change = std::fmod(std::fabs(my.angle - dbg_last_my_angle_), 360.0); if(angle_change > 180.0) angle_change = 360.0 - angle_change;
        spin_pen = (move_dist < 1.0 && angle_change > 5.0) ? -0.02 : 0.0;
        step_cost = -0.001;
        double r = dist_prog + align_prog + spin_pen + step_cost;

        dbg_last_my_pos_ = my.pos; dbg_last_my_angle_ = my.angle; dbg_last_enemy_pos_ = en.pos;
        dbg_prev_dist_norm_ = dist_norm; dbg_prev_align_norm_ = align_norm;
        return r;
    }
}
