#include "rl/TankEnv.h"
#include "util/Math.h"
#include "Shell.h"
#include <cmath>
#include <algorithm>

namespace TankTrouble
{
    static bool pointInRect(double rectAngle, const util::Vec &center, double w, double h, const util::Vec &p)
    {
        auto axis = util::getUnitVectors(rectAngle);
        util::Vec d = util::Vec(p.x() - center.x(), p.y() - center.y());
        double xproj = d.x() * axis.first.x() + d.y() * axis.first.y();
        double yproj = d.x() * axis.second.x() + d.y() * axis.second.y();
        return std::fabs(xproj) <= w / 2.0 && std::fabs(yproj) <= h / 2.0;
    }

    TankEnv::TankEnv() : agent_tank_id_(PLAYER_TANK_ID), enemy_tank_id_(AI_TANK_ID)
    {
        rl_controller_ = std::make_unique<RLController>();
        printf("[DEBUG] TankEnv(): RLController created.\n");
        rl_controller_->start();
        printf("[DEBUG] TankEnv(): RLController started.\n");
    }
    TankEnv::TankEnv(RLController *rlc) : agent_tank_id_(PLAYER_TANK_ID), enemy_tank_id_(AI_TANK_ID)
    {
        rl_controller_.reset();
        if (rlc)
            rl_controller_.reset(rlc);
    }

    bool TankEnv::hasDirectLineToEnemy(const util::Vec &my_pos, const util::Vec &enemy_pos)
    {
        // Check if there's a direct line of sight from my tank to enemy tank
        // by ray-marching from my position to enemy position
        double dx = enemy_pos.x() - my_pos.x();
        double dy = enemy_pos.y() - my_pos.y();
        double distance = std::hypot(dx, dy);

        if (distance < 1.0)
            return true; // Too close, consider it direct

        dx /= distance; // normalize
        dy /= distance;

        auto *blocks = rl_controller_->getBlocks();
        const double step = 5.0; // pixels per step

        for (double t = 0.0; t < distance; t += step)
        {
            util::Vec p(my_pos.x() + dx * t, my_pos.y() + dy * t);

            // Check border
            if (p.x() <= 0 || p.x() >= GAME_VIEW_WIDTH || p.y() <= 0 || p.y() >= GAME_VIEW_HEIGHT)
                return false;

            // Check blocks
            for (const auto &kv : *blocks)
            {
                const Block &b = kv.second;
                auto bc = b.center();
                // Simple AABB check
                if (std::abs(p.x() - bc.x()) < b.width() / 2 && std::abs(p.y() - bc.y()) < b.height() / 2)
                {
                    return false; // Wall blocks the line
                }
            }
        }

        return true; // No obstacles found
    }

    std::vector<double> TankEnv::reset()
    {
        // reset前先安全退出RL线程
        if (rl_controller_)
            rl_controller_->quitGame();
        rl_controller_ = std::make_unique<RLController>();
        rl_controller_->start();
        // 查找新坦克ID
        auto objsPtr = rl_controller_->getObjects();
        auto &objs = *objsPtr;
        agent_tank_id_ = -1;
        enemy_tank_id_ = -1;
        for (const auto &kv : objs)
        {
            if (kv.second->type() == OBJ_TANK)
            {
                Tank *tank = dynamic_cast<Tank *>(kv.second.get());
                if (tank)
                {
                    if (agent_tank_id_ == -1)
                        agent_tank_id_ = tank->id();
                    else
                        enemy_tank_id_ = tank->id();
                }
            }
        }
        // Reset tracking variables
        last_me_alive_ = true;
        my_bullet_ids_.clear();
        prev_closest_own_bullet_dist_ = 1000.0;

        // initialize reward shaping memory
        if (objs.find(agent_tank_id_) != objs.end() && objs.find(enemy_tank_id_) != objs.end())
        {
            auto *me = dynamic_cast<Tank *>(objs[agent_tank_id_].get());
            auto *enemy = dynamic_cast<Tank *>(objs[enemy_tank_id_].get());
            auto my = me->getCurrentPosition();
            auto en = enemy->getCurrentPosition();
            last_my_pos_ = my.pos;
            last_my_angle_ = my.angle;
            last_enemy_pos_ = en.pos;
            double dist = std::hypot(en.pos.x() - my.pos.x(), en.pos.y() - my.pos.y());
            double maxd = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
            prev_dist_norm_ = dist / maxd;
            double bearing = std::atan2(en.pos.y() - my.pos.y(), en.pos.x() - my.pos.x()) * 180.0 / M_PI;
            double diff = std::fmod(std::fabs(bearing - my.angle), 360.0);
            if (diff > 180.0)
                diff = 360.0 - diff;
            prev_align_norm_ = diff / 180.0;

            // 用欧氏距离初始化 prev_path_dist_norm_
            double path_dist = std::hypot(en.pos.x() - my.pos.x(), en.pos.y() - my.pos.y());
            double max_path_dist = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
            prev_path_dist_norm_ = path_dist / max_path_dist;
        }
        return getCurrentState();
    }

    std::tuple<std::vector<double>, double, bool> TankEnv::step(int action)
    {
        // Record bullet state before step (to detect if we're hit by own bullet)
        auto objsPtr_before = rl_controller_->getObjects();
        auto &objs_before = *objsPtr_before;
        std::vector<int> bullets_before_agent;
        std::vector<int> bullets_before_enemy;
        bool me_alive_before = objs_before.find(agent_tank_id_) != objs_before.end();

        if (me_alive_before)
        {
            auto *me_before = dynamic_cast<Tank *>(objs_before[agent_tank_id_].get());
            auto my_before = me_before->getCurrentPosition();

            for (auto &kv : objs_before)
            {
                if (kv.second->type() == OBJ_SHELL)
                {
                    Shell *sh = dynamic_cast<Shell *>(kv.second.get());
                    auto sh_pos = sh->getCurrentPosition();
                    double d = std::hypot(sh_pos.pos.x() - my_before.pos.x(), sh_pos.pos.y() - my_before.pos.y());

                    if (sh->tankId() == agent_tank_id_ && d < 100.0)
                    {
                        bullets_before_agent.push_back(sh->id());
                    }
                    else if (sh->tankId() == enemy_tank_id_ && d < 100.0)
                    {
                        bullets_before_enemy.push_back(sh->id());
                    }
                }
            }
        }

        applyActionToAgent(action);
        // advance a fixed number of ticks to simulate one step
        for (int i = 0; i < 5; i++)
            rl_controller_->stepOnce();
        step_counter_++; // Increment step counter for timing

        // Check if agent died and track which bullets disappeared
        auto objsPtr_after = rl_controller_->getObjects();
        auto &objs_after = *objsPtr_after;
        bool me_alive_after = objs_after.find(agent_tank_id_) != objs_after.end();

        // Track if agent's own bullets disappeared (might have hit agent)
        bool own_bullet_hit = false;
        if (me_alive_before && !me_alive_after)
        {
            // Agent just died, check if own bullets disappeared
            for (int bullet_id : bullets_before_agent)
            {
                if (objs_after.find(bullet_id) == objs_after.end())
                {
                    // Agent's bullet disappeared right when agent died
                    own_bullet_hit = true;
                    break;
                }
            }
        }

        bool done = false;
        double reward = calculateReward(done, own_bullet_hit);
        std::vector<double> ns = getCurrentState();
        return {ns, reward, done};
    }

    int TankEnv::getSmithAction()
    {
        if (!rl_controller_)
        {
            printf("RLController not initialized in getSmithAction()\n");
            return 0;
        }
        return rl_controller_->getSmithAction();
    }

    int TankEnv::getAgentSmithAction()
    {
        if (!rl_controller_)
        {
            printf("RLController not initialized in getAgentSmithAction()\n");
            return 0;
        }
        return rl_controller_->getAgentSmithAction();
    }

    void TankEnv::applyActionToAgent(int action)
    {
        switch (action)
        {
        case DO_NOTHING:
        {
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::StopForward));
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::StopBackward));
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::StopRotateCW));
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::StopRotateCCW));
            break;
        }
        case MOVE_FORWARD:
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::Forward));
            break;
        case MOVE_BACKWARD:
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::Backward));
            break;
        case ROTATE_CW:
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::RotateCW));
            break;
        case ROTATE_CCW:
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::RotateCCW));
            break;
        case SHOOT:
            rl_controller_->dispatchEvent(ControlEvent(ControlEvent::Fire));
            break;
        default:
            break;
        }
    }

    double TankEnv::normalizeX(double x) { return x / static_cast<double>(GAME_VIEW_WIDTH); }
    double TankEnv::normalizeY(double y) { return y / static_cast<double>(GAME_VIEW_HEIGHT); }

    void TankEnv::angleToSinCos(double angleDeg, double &s, double &c)
    {
        double rad = angleDeg * M_PI / 180.0;
        s = std::sin(rad);
        c = std::cos(rad);
    }

    std::vector<double> TankEnv::getCurrentState()
    {
        std::vector<double> state;
        auto objsPtr = rl_controller_->getObjects();
        auto &objs = *objsPtr;
        if (objs.find(agent_tank_id_) == objs.end() || objs.find(enemy_tank_id_) == objs.end())
        {
            return std::vector<double>(9 + 1 + 72, 0.0);
        }
        auto *me = dynamic_cast<Tank *>(objs[agent_tank_id_].get());
        auto *enemy = dynamic_cast<Tank *>(objs[enemy_tank_id_].get());
        Object::PosInfo my = me->getCurrentPosition();
        Object::PosInfo en = enemy->getCurrentPosition();

        state.push_back(normalizeX(my.pos.x()));
        state.push_back(normalizeY(my.pos.y()));
        double s, c;
        angleToSinCos(my.angle, s, c);
        state.push_back(s);
        state.push_back(c);
        state.push_back(me->remainShells() > 0 ? 1.0 : 0.0);

        state.push_back(normalizeX(en.pos.x() - my.pos.x()));
        state.push_back(normalizeY(en.pos.y() - my.pos.y()));
        double se, ce;
        angleToSinCos(en.angle, se, ce);
        state.push_back(se);
        state.push_back(ce);

        // ==================== 新增：直线视线标志 ====================
        // 添加一个特征：是否有到敌人的直线视线（0.0 或 1.0）
        state.push_back(hasDirectLineToEnemy(my.pos, en.pos) ? 1.0 : 0.0);

        // 使用24条射线，每条3维
        std::vector<double> rays = rayFeatures(24);
        state.insert(state.end(), rays.begin(), rays.end());

        return state;
    }

    std::vector<double> TankEnv::rayFeatures(int num_rays)
    {
        std::vector<double> feats;
        auto objsPtr = rl_controller_->getObjects();
        auto &objs = *objsPtr;
        if (objs.find(agent_tank_id_) == objs.end())
            return std::vector<double>(num_rays * 3, 1.0);
        auto *me = dynamic_cast<Tank *>(objs[agent_tank_id_].get());
        Object::PosInfo my = me->getCurrentPosition();
        const double MAX_DIST = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        auto *blocks = rl_controller_->getBlocks();

        for (int i = 0; i < num_rays; i++)
        {
            double ang = (360.0 / num_rays) * i;
            double rad = ang * M_PI / 180.0;
            double dx = std::cos(rad);
            double dy = std::sin(rad);
            double step = 4.0; // pixels per march
            double wallD = MAX_DIST, enemyD = MAX_DIST, bulletD = MAX_DIST;
            for (double t = 0.0; t <= MAX_DIST; t += step)
            {
                util::Vec p(my.pos.x() + dx * t, my.pos.y() + dy * t);
                // borders
                if (p.x() <= 0 || p.x() >= GAME_VIEW_WIDTH || p.y() <= 0 || p.y() >= GAME_VIEW_HEIGHT)
                {
                    wallD = t;
                    break;
                }
                // blocks
                for (const auto &kv : *blocks)
                {
                    const Block &b = kv.second;
                    double bAng = b.isHorizon() ? 0.0 : 90.0;
                    if (pointInRect(bAng, b.center(), b.width(), b.height(), p))
                    {
                        wallD = t;
                    }
                }
                // objects
                for (auto &kv : objs)
                {
                    Object *obj = kv.second.get();
                    if (obj->type() == OBJ_TANK)
                    {
                        Tank *tnk = dynamic_cast<Tank *>(obj);
                        auto pos = tnk->getCurrentPosition();
                        if (pointInRect(pos.angle, pos.pos, Tank::TANK_WIDTH, Tank::TANK_HEIGHT, p))
                        {
                            double d = t;
                            if (kv.first == enemy_tank_id_)
                                enemyD = std::min(enemyD, d);
                        }
                    }
                    else if (obj->type() == OBJ_SHELL)
                    {
                        Shell *sh = dynamic_cast<Shell *>(obj);
                        auto pos = sh->getCurrentPosition();
                        double dist = std::hypot(p.x() - pos.pos.x(), p.y() - pos.pos.y());
                        if (dist <= Shell::RADIUS)
                            bulletD = std::min(bulletD, t);
                    }
                }
                if (wallD < MAX_DIST)
                    break;
            }
            feats.push_back(wallD / MAX_DIST);
            feats.push_back(enemyD < MAX_DIST ? enemyD / MAX_DIST : 1.0);
            feats.push_back(bulletD < MAX_DIST ? bulletD / MAX_DIST : 1.0);
        }
        return feats;
    }

    double TankEnv::calculateReward(bool &done, bool killed_by_own_bullet)
    {
        done = false;
        auto objsPtr = rl_controller_->getObjects();
        auto &objs = *objsPtr;
        bool meAlive = objs.find(agent_tank_id_) != objs.end();
        bool enemyAlive = objs.find(enemy_tank_id_) != objs.end();

        // 终止奖励：只有杀死敌人是正奖励
        if (!meAlive || !enemyAlive)
        {
            done = true;
            if (meAlive && !enemyAlive)
                return 100.0; // 唯一正奖励
            if (!meAlive && enemyAlive)
                return -100.0; // 死亡大惩罚
            return -50.0;      // 平局也惩罚
        }

        // 其余所有情况都为负奖励（但区分惩罚大小）
        double r = 0.0;

        // 获取智能体和敌人的状态
        auto *me = dynamic_cast<Tank *>(objs[agent_tank_id_].get());
        auto *enemy = dynamic_cast<Tank *>(objs[enemy_tank_id_].get());
        auto my = me->getCurrentPosition();
        auto en = enemy->getCurrentPosition();

        // 移动惩罚：不动-1，动-0.5
        double move_dist = std::hypot(my.pos.x() - last_my_pos_.x(), my.pos.y() - last_my_pos_.y());
        if (move_dist < 0.5)
            r -= 1.0;
        else
            r -= 0.5;

        // 靠近子弹惩罚已取消

        // 射击惩罚
        int current_shells = me->remainShells();
        if (current_shells < last_my_shells_)
            r -= 1.0;
        last_my_shells_ = current_shells;

        // 旋转惩罚
        double angle_change = std::fmod(std::fabs(my.angle - last_my_angle_), 360.0);
        if (angle_change > 180.0)
            angle_change = 360.0 - angle_change;
        if (angle_change > 30.0)
            r -= 0.2;

        // 更新记忆
        last_my_pos_ = my.pos;
        last_my_angle_ = my.angle;
        last_enemy_pos_ = en.pos;

        return r;
    }

}
