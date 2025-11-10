#ifndef TANK_TROUBLE_ATTACK_STRATEGY_H
#define TANK_TROUBLE_ATTACK_STRATEGY_H
#include "Strategy.h"
#include "Object.h"

namespace TankTrouble
{
    class AttackStrategy : public Strategy
    {
    public:
        explicit AttackStrategy(const Object::PosInfo &pos) : Strategy(Strategy::Attack),
                                                              attackingPos(pos),
                                                              done(false) {}

        bool update(LocalController *ctl, Tank *tank, uint64_t globalStep, AgentSmith *agent) override;
        void cancelAttack();

    private:
        Object::PosInfo attackingPos;
        bool done;
    };
} // namespace TankTrouble

#endif // TANK_TROUBLE_ATTACK_STRATEGY_H