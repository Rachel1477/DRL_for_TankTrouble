//
// Reinforcement Learning Environment for TankTrouble (local-only)
//

#ifndef TANK_TROUBLE_TANK_ENV_H
#define TANK_TROUBLE_TANK_ENV_H

#include <memory>
#include <vector>
#include <tuple>
#include <deque>
#include "controller/LocalController.h"

namespace TankTrouble
{
    class TankEnv
    {
    public:
        enum Action
        {
            DO_NOTHING = 0,
            MOVE_FORWARD = 1,
            MOVE_BACKWARD = 2,
            ROTATE_CW = 3,
            ROTATE_CCW = 4,
            SHOOT = 5
        };

        TankEnv();

        std::vector<double> reset();

        std::tuple<std::vector<double>, double, bool> step(int action);

    private:
        std::unique_ptr<LocalController> controller_;
        int agent_tank_id_;
        int enemy_tank_id_;

        void applyActionToAgent(int action);
        std::vector<double> getCurrentState();
        double calculateReward(bool& done);

        // helpers
        static double normalizeX(double x);
        static double normalizeY(double y);
        static void angleToSinCos(double angleDeg, double& s, double& c);
        std::vector<double> rayFeatures();

        // reward shaping state
        util::Vec last_my_pos_{};
        double last_my_angle_ = 0.0;
        util::Vec last_enemy_pos_{};
        double prev_dist_norm_ = 0.0;    // distance to enemy normalized
        double prev_align_norm_ = 0.0;   // alignment error normalized [0,1]
        int last_my_shells_ = 3;         // track shooting
        double prev_closest_bullet_dist_ = 1000.0;  // track bullet avoidance
        
        // New: direct line-of-sight shooting reward
        double last_direct_shot_time_ = -100.0;  // timestamp of last direct shot reward
        
        // New: rapid fire penalty tracking (3 shots in 3 seconds)
        std::deque<double> recent_shot_times_;   // timestamps of recent shots
        
        // Helper methods
        bool hasDirectLineToEnemy(const util::Vec& my_pos, const util::Vec& enemy_pos);
        
        // Simple step counter for timing (instead of globalSteps)
        mutable int step_counter_ = 0;
    };
}

#endif // TANK_TROUBLE_TANK_ENV_H


