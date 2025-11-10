//
// RL Controller for Agent vs SmithAI training mode
// This controller runs training loop and displays it in GUI
//

#ifndef TANK_TROUBLE_RL_CONTROLLER_H
#define TANK_TROUBLE_RL_CONTROLLER_H

#include "LocalController.h"
#include "Controller.h"
#include "rl/TankEnv.h"
#include <memory>
#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <chrono>

namespace TankTrouble
{
    class TankEnv;
    class RLController : public Controller
    {
    public:
        typedef std::function<int(const std::vector<double> &)> GetActionCallback;
        typedef std::function<void(int, double, bool)> EpisodeEndCallback;
        typedef std::function<void(const std::vector<double> &, int, double, const std::vector<double> &, bool)> StepCallback;

        RLController(LocalController *shared = nullptr);
        ~RLController();

        void setGetActionCallback(GetActionCallback cb) { get_action_cb_ = cb; }
        void setEpisodeEndCallback(EpisodeEndCallback cb) { episode_end_cb_ = cb; }
        void setStepCallback(StepCallback cb) { step_cb_ = cb; }
        int getEpisodeCount() const { return episode_count_; }
        double getTotalReward() const { return total_reward_; }

        static RLController *getGlobalInstance();

        // ! Copy The LocalController
        void start() override;
        void quitGame() override;
        void dispatchEvent(const ControlEvent &event) override;
        void resetImmediate();
        void stepOnce();
        int getSmithAction();
        int getAgentSmithAction();
        // ! Forward Controller's data accessors for GUI
        ObjectListPtr getObjects() override;
        BlockList *getBlocks() override;
        std::vector<PlayerInfo> getPlaysInfo() override;

    private:
        GetActionCallback get_action_cb_;
        EpisodeEndCallback episode_end_cb_;
        StepCallback step_cb_;

        std::atomic<bool> training_active_{false};
        int episode_count_;
        double total_reward_;

        LocalController *ctl_shared_ = nullptr;
        bool own_controller_ = false;
        std::thread agent_thread_;
        std::thread episode_thread_;
        bool use_python_cb_ = false;

        void agentLoop();
        int decideActionFallback();
        void applyAgentAction(int action);
        void episodeLoop();
        std::vector<double> buildState();

        util::Vec dbg_last_my_pos_{};
        double dbg_last_my_angle_ = 0.0;
        util::Vec dbg_last_enemy_pos_{};
        double dbg_prev_dist_norm_ = 0.0;
        double dbg_prev_align_norm_ = 0.0;
        bool dbg_initialized_ = false;
        void dbgInitMemory();
        double dbgComputeReward(double &dist_prog, double &align_prog, double &spin_pen, double &step_cost);
    };
}

#endif // TANK_TROUBLE_RL_CONTROLLER_H
