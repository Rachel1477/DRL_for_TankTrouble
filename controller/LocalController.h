//
// Created by zr on 23-2-8.
//

#ifndef TANK_TROUBLE_LOCAL_CONTROLLER_H
#define TANK_TROUBLE_LOCAL_CONTROLLER_H
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <queue>
#include <chrono>
#include "Controller.h"
#include "Object.h"
#include "util/Vec.h"
#include "Maze.h"
#include "defs.h"
#include "smithAI/DodgeStrategy.h"
#include "smithAI/ContactStrategy.h"
#include "smithAI/AttackStrategy.h"
#include "smithAI/Strategy.h"
#include "smithAI/AStar.h"
#include "Block.h"

#define UPWARDS 0
#define UPWARDS_LEFT 1
#define LEFT 2
#define DOWNWARDS_LEFT 3
#define DOWNWARDS 4
#define DOWNWARDS_RIGHT 5
#define RIGHT 6
#define UPWARDS_RIGHT 7

#define PLAYER_TANK_ID 1
#define AI_TANK_ID 2

namespace TankTrouble
{
    class AgentSmith;
    class DodgeStrategy;
    class ContactStrategy;
    class AttackStrategy;

    class LocalController : public Controller
    {
    public:
        LocalController();
        ~LocalController() override;
        void start() override;
        void dispatchEvent(const ControlEvent &event) override;
        void quitGame() override;

        // Methods for RL environment (synchronous stepping)
        ObjectListPtr getObjects() override;
        BlockList *getBlocks() override;
        std::vector<PlayerInfo> getPlaysInfo() override;
        void resetImmediate();
        void stepOnce();
        // Return AgentSmith's chosen discrete action for the AI tank
        // Values correspond to TankEnv::Action enum
        int getSmithAction();
        // Return AgentSmith's chosen discrete action for the AGENT tank (agent-perspective)
        int getRLSmithAction();

    private:
        void restart(double delay);
        void initAll();
        void resetState();
        void run();
        void moveAll();
        void controlEventHandler(const ControlEvent &event);
        void updateStrategy(Strategy *strategy, AgentSmith *agent);
        void fire(Tank *tank);

        int checkShellCollision(const Object::PosInfo &curPos, const Object::PosInfo &nextPos);
        int checkShellBlockCollision(const Object::PosInfo &curPos, const Object::PosInfo &nextPos);
        int checkShellTankCollision(const Object::PosInfo &curPos, const Object::PosInfo &nextPos);

        int checkTankBlockCollision(const Object::PosInfo &curPos, const Object::PosInfo &nextPos);
        Object::PosInfo getBouncedPosition(const Object::PosInfo &cur, const Object::PosInfo &next, int blockId);

        void initBlocks();
        struct PairHash
        {
            template <typename T1, typename T2>
            size_t operator()(const std::pair<T1, T2> &p) const
            {
                return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
            }
        };
        static std::vector<Object::PosInfo> getRandomPositions(int num);
        bool getSmithPosition(Object::PosInfo &pos);
        bool getMyPosition(Object::PosInfo &pos);

        Maze maze;
        std::vector<int> deletedObjs;
        ObjectList objects;
        std::vector<int> shellPossibleCollisionBlocks[HORIZON_GRID_NUMBER][VERTICAL_GRID_NUMBER][8];
        std::vector<int> tankPossibleCollisionBlocks[HORIZON_GRID_NUMBER][VERTICAL_GRID_NUMBER];
        uint64_t globalSteps;

        std::mutex eventsMu;
        std::queue<ControlEvent> pendingEvents;
        std::atomic<bool> quitting{false};
        std::atomic<bool> pendingRestart{false};
        std::chrono::steady_clock::time_point restartAt;

        friend class AgentSmith;
        friend class DodgeStrategy;
        friend class ContactStrategy;
        friend class AttackStrategy;
        std::unique_ptr<AgentSmith> agent_smith; // for AI_TANK_ID (enemy)
        std::unique_ptr<AgentSmith> rl_smith_ai; // for RL agent (PLAYER_TANK_ID)
        int danger;
        std::unique_ptr<DodgeStrategy> smithDodgeStrategy;
        std::unique_ptr<ContactStrategy> smithContactStrategy;
        std::unique_ptr<AttackStrategy> smithAttackStrategy;
        // rl_smith_ai 专用策略
        int rl_danger;
        std::unique_ptr<DodgeStrategy> rlSmithDodgeStrategy;
        std::unique_ptr<ContactStrategy> rlSmithContactStrategy;
        std::unique_ptr<AttackStrategy> rlSmithAttackStrategy;
    };
}

#endif // TANK_TROUBLE_LOCAL_CONTROLLER_H
