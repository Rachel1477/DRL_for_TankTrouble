//
// Reinforcement Learning Environment for TankTrouble (local-only)
//

#ifndef TANK_TROUBLE_TANK_ENV_H
#define TANK_TROUBLE_TANK_ENV_H

#include <memory>
#include <vector>
#include <tuple>
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
    };
}

#endif // TANK_TROUBLE_TANK_ENV_H


