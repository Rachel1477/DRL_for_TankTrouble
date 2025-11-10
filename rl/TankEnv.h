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
#include "controller/RLController.h"
#include "smithAI/AStar.h"
#include "defs.h"

namespace TankTrouble
{
    class RLController;
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

        TankEnv(RLController *rlc);

        std::vector<double> reset();

        std::tuple<std::vector<double>, double, bool> step(int action);

        // Query AgentSmith's discrete action for the AI tank
        int getSmithAction();
        // Query AgentSmith's discrete action for the AGENT tank (agent-perspective)
        int getAgentSmithAction();

    public:
        std::unique_ptr<RLController> rl_controller_;
        int agent_tank_id_;
        int enemy_tank_id_;
        // std::unique_ptr<AStar> astar_; // 移除AStar

        void applyActionToAgent(int action);
        std::vector<double> getCurrentState();
        double calculateReward(bool &done, bool killed_by_own_bullet = false);

        // helpers
        static double normalizeX(double x);
        static double normalizeY(double y);
        static void angleToSinCos(double angleDeg, double &s, double &c);
        std::vector<double> rayFeatures(int num_rays);

        // Pathfinding helpers
        // double getPathDistance(const util::Vec &my_pos, const util::Vec &enemy_pos); // 移除AStar
        // util::Vec getNextPathPoint(const util::Vec &my_pos, const util::Vec &enemy_pos); // 移除AStar
        // void initializeAStar(); // 移除AStar

        // reward shaping state
        util::Vec last_my_pos_{};
        double last_my_angle_ = 0.0;
        util::Vec last_enemy_pos_{};
        double prev_dist_norm_ = 0.0;                  // distance to enemy normalized
        double prev_align_norm_ = 0.0;                 // alignment error normalized [0,1]
        int last_my_shells_ = 3;                       // track shooting
        double prev_closest_bullet_dist_ = 1000.0;     // track bullet avoidance
        double prev_closest_own_bullet_dist_ = 1000.0; // track own bullet distance
        double prev_path_dist_norm_ = 1.0;             // previous path distance (normalized)
        bool last_me_alive_ = true;                    // track agent's life status
        std::vector<int> my_bullet_ids_;               // track bullets fired by agent

        // Helper methods
        bool hasDirectLineToEnemy(const util::Vec &my_pos, const util::Vec &enemy_pos);

        // Simple step counter for timing (instead of globalSteps)
        mutable int step_counter_ = 0;
    };
}

#endif // TANK_TROUBLE_TANK_ENV_H
