//
// RL Controller for Agent vs SmithAI training mode
// This controller runs training loop and displays it in GUI
//

#ifndef TANK_TROUBLE_RL_CONTROLLER_H
#define TANK_TROUBLE_RL_CONTROLLER_H

#include "LocalController.h"
#include <memory>
#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <chrono>

namespace TankTrouble
{
    class RLController : public LocalController
    {
    public:
        // Callback function types for Python training
        typedef std::function<int(const std::vector<double>&)> GetActionCallback;
        typedef std::function<void(int, double, bool)> EpisodeEndCallback;
        typedef std::function<void(const std::vector<double>&, int, double, const std::vector<double>&, bool)> StepCallback;

        RLController();
        ~RLController() override;

        void start() override;
        void quitGame() override;

        void setGetActionCallback(GetActionCallback cb) { get_action_cb_ = cb; }
        void setEpisodeEndCallback(EpisodeEndCallback cb) { episode_end_cb_ = cb; }
        void setStepCallback(StepCallback cb) { step_cb_ = cb; }

        int getEpisodeCount() const { return episode_count_; }
        double getTotalReward() const { return total_reward_; }

    private:
        GetActionCallback get_action_cb_;
        EpisodeEndCallback episode_end_cb_;
        StepCallback step_cb_;

        std::atomic<bool> training_active_{false};
        int episode_count_;
        double total_reward_;

        // Agent decision thread
        std::thread agent_thread_;
        std::thread episode_thread_;
        void agentLoop();
        int decideActionFallback();
        void applyAgentAction(int action);
        void episodeLoop();
        std::vector<double> buildState();

        // debug / shaping for GUI path
        bool use_python_cb_ = false;
        util::Vec dbg_last_my_pos_{}; double dbg_last_my_angle_ = 0.0;
        util::Vec dbg_last_enemy_pos_{}; double dbg_prev_dist_norm_ = 0.0; double dbg_prev_align_norm_ = 0.0;
        bool dbg_initialized_ = false;
        void dbgInitMemory();
        double dbgComputeReward(double& dist_prog, double& align_prog, double& spin_pen, double& step_cost);
    };
}

#endif // TANK_TROUBLE_RL_CONTROLLER_H

