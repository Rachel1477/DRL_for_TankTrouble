#include "rl/TankEnv.h"
#include "util/Math.h"
#include "Shell.h"
#include <cmath>

namespace TankTrouble
{
    static bool pointInRect(double rectAngle, const util::Vec& center, double w, double h, const util::Vec& p)
    {
        auto axis = util::getUnitVectors(rectAngle);
        util::Vec d = util::Vec(p.x() - center.x(), p.y() - center.y());
        double xproj = d.x() * axis.first.x() + d.y() * axis.first.y();
        double yproj = d.x() * axis.second.x() + d.y() * axis.second.y();
        return std::fabs(xproj) <= w / 2.0 && std::fabs(yproj) <= h / 2.0;
    }

    TankEnv::TankEnv(): controller_(new LocalController()), agent_tank_id_(PLAYER_TANK_ID), enemy_tank_id_(AI_TANK_ID)
    {
        // no GUI, no controller->start(); we step synchronously
    }

    std::vector<double> TankEnv::reset()
    {
        controller_->resetImmediate();
        return getCurrentState();
    }

    std::tuple<std::vector<double>, double, bool> TankEnv::step(int action)
    {
        applyActionToAgent(action);
        // advance a fixed number of ticks to simulate one step
        for(int i = 0; i < 5; i++) controller_->stepOnce();
        bool done = false;
        double reward = calculateReward(done);
        std::vector<double> ns = getCurrentState();
        return {ns, reward, done};
    }

    void TankEnv::applyActionToAgent(int action)
    {
        switch(action)
        {
            case DO_NOTHING:
            {
                controller_->dispatchEvent(ControlEvent(ControlEvent::StopForward));
                controller_->dispatchEvent(ControlEvent(ControlEvent::StopBackward));
                controller_->dispatchEvent(ControlEvent(ControlEvent::StopRotateCW));
                controller_->dispatchEvent(ControlEvent(ControlEvent::StopRotateCCW));
                break;
            }
            case MOVE_FORWARD:
                controller_->dispatchEvent(ControlEvent(ControlEvent::Forward));
                break;
            case MOVE_BACKWARD:
                controller_->dispatchEvent(ControlEvent(ControlEvent::Backward));
                break;
            case ROTATE_CW:
                controller_->dispatchEvent(ControlEvent(ControlEvent::RotateCW));
                break;
            case ROTATE_CCW:
                controller_->dispatchEvent(ControlEvent(ControlEvent::RotateCCW));
                break;
            case SHOOT:
                controller_->dispatchEvent(ControlEvent(ControlEvent::Fire));
                break;
            default: break;
        }
    }

    double TankEnv::normalizeX(double x) { return x / static_cast<double>(GAME_VIEW_WIDTH); }
    double TankEnv::normalizeY(double y) { return y / static_cast<double>(GAME_VIEW_HEIGHT); }

    void TankEnv::angleToSinCos(double angleDeg, double& s, double& c)
    {
        double rad = angleDeg * M_PI / 180.0;
        s = std::sin(rad); c = std::cos(rad);
    }

    std::vector<double> TankEnv::getCurrentState()
    {
        std::vector<double> state;
        auto objsPtr = controller_->getObjects();
        auto& objs = *objsPtr;
        if(objs.find(agent_tank_id_) == objs.end() || objs.find(enemy_tank_id_) == objs.end())
        {
            // terminal: return zero vector
            return std::vector<double>(8 + 16 * 3, 0.0);
        }
        auto* me = dynamic_cast<Tank*>(objs[agent_tank_id_].get());
        auto* enemy = dynamic_cast<Tank*>(objs[enemy_tank_id_].get());
        Object::PosInfo my = me->getCurrentPosition();
        Object::PosInfo en = enemy->getCurrentPosition();

        state.push_back(normalizeX(my.pos.x()));
        state.push_back(normalizeY(my.pos.y()));
        double s, c; angleToSinCos(my.angle, s, c);
        state.push_back(s); state.push_back(c);
        state.push_back(me->remainShells() > 0 ? 1.0 : 0.0);

        state.push_back(normalizeX(en.pos.x() - my.pos.x()));
        state.push_back(normalizeY(en.pos.y() - my.pos.y()));
        double se, ce; angleToSinCos(en.angle, se, ce);
        state.push_back(se); state.push_back(ce);

        std::vector<double> rays = rayFeatures();
        state.insert(state.end(), rays.begin(), rays.end());
        return state;
    }

    std::vector<double> TankEnv::rayFeatures()
    {
        std::vector<double> feats;
        auto objsPtr = controller_->getObjects();
        auto& objs = *objsPtr;
        if(objs.find(agent_tank_id_) == objs.end()) return std::vector<double>(16 * 3, 1.0);
        auto* me = dynamic_cast<Tank*>(objs[agent_tank_id_].get());
        Object::PosInfo my = me->getCurrentPosition();
        const int NUM_RAYS = 16;
        const double MAX_DIST = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        auto* blocks = controller_->getBlocks();

        for(int i = 0; i < NUM_RAYS; i++)
        {
            double ang = (360.0 / NUM_RAYS) * i;
            double rad = ang * M_PI / 180.0;
            double dx = std::cos(rad);
            double dy = std::sin(rad);
            double step = 4.0; // pixels per march
            double wallD = MAX_DIST, enemyD = MAX_DIST, bulletD = MAX_DIST;
            for(double t = 0.0; t <= MAX_DIST; t += step)
            {
                util::Vec p(my.pos.x() + dx * t, my.pos.y() + dy * t);
                // borders
                if(p.x() <= 0 || p.x() >= GAME_VIEW_WIDTH || p.y() <= 0 || p.y() >= GAME_VIEW_HEIGHT)
                {
                    wallD = t; break;
                }
                // blocks
                for(const auto& kv : *blocks)
                {
                    const Block& b = kv.second;
                    double bAng = b.isHorizon() ? 0.0 : 90.0;
                    if(pointInRect(bAng, b.center(), b.width(), b.height(), p))
                    { wallD = t; }
                }
                // objects
                for(auto& kv : objs)
                {
                    Object* obj = kv.second.get();
                    if(obj->type() == OBJ_TANK)
                    {
                        Tank* tnk = dynamic_cast<Tank*>(obj);
                        auto pos = tnk->getCurrentPosition();
                        if(pointInRect(pos.angle, pos.pos, Tank::TANK_WIDTH, Tank::TANK_HEIGHT, p))
                        {
                            double d = t; if(kv.first == enemy_tank_id_) enemyD = std::min(enemyD, d);
                        }
                    }
                    else if(obj->type() == OBJ_SHELL)
                    {
                        Shell* sh = dynamic_cast<Shell*>(obj);
                        auto pos = sh->getCurrentPosition();
                        double dist = std::hypot(p.x() - pos.pos.x(), p.y() - pos.pos.y());
                        if(dist <= Shell::RADIUS) bulletD = std::min(bulletD, t);
                    }
                }
                if(wallD < MAX_DIST) break;
            }
            feats.push_back(wallD / MAX_DIST);
            feats.push_back(enemyD < MAX_DIST ? enemyD / MAX_DIST : 1.0);
            feats.push_back(bulletD < MAX_DIST ? bulletD / MAX_DIST : 1.0);
        }
        return feats;
    }

    double TankEnv::calculateReward(bool& done)
    {
        done = false;
        auto objsPtr = controller_->getObjects();
        auto& objs = *objsPtr;
        bool meAlive = objs.find(agent_tank_id_) != objs.end();
        bool enemyAlive = objs.find(enemy_tank_id_) != objs.end();
        if(!meAlive || !enemyAlive)
        {
            done = true;
            if(meAlive && !enemyAlive) return 100.0;
            if(!meAlive && enemyAlive) return -100.0;
            return 0.0;
        }
        // shaping: living reward
        return 0.01;
    }
}


