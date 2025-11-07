#include "rl/TankEnv.h"
#include "util/Math.h"
#include "Shell.h"
#include <cmath>

namespace TankTrouble {
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
    
    
    bool TankEnv::hasDirectLineToEnemy(const util::Vec& my_pos, const util::Vec& enemy_pos)
    {
        // Check if there's a direct line of sight from my tank to enemy tank
        // by ray-marching from my position to enemy position
        double dx = enemy_pos.x() - my_pos.x();
        double dy = enemy_pos.y() - my_pos.y();
        double distance = std::hypot(dx, dy);
        
        if (distance < 1.0) return true;  // Too close, consider it direct
        
        dx /= distance;  // normalize
        dy /= distance;
        
        auto* blocks = controller_->getBlocks();
        const double step = 5.0;  // pixels per step
        
        for (double t = 0.0; t < distance; t += step)
        {
            util::Vec p(my_pos.x() + dx * t, my_pos.y() + dy * t);
            
            // Check border
            if (p.x() <= 0 || p.x() >= GAME_VIEW_WIDTH || p.y() <= 0 || p.y() >= GAME_VIEW_HEIGHT)
                return false;
            
            // Check blocks
            for (const auto& kv : *blocks)
            {
                const Block& b = kv.second;
                auto bc = b.center();
                // Simple AABB check
                if (std::abs(p.x() - bc.x()) < b.width()/2 && std::abs(p.y() - bc.y()) < b.height()/2)
                {
                    return false;  // Wall blocks the line
                }
            }
        }
        
        return true;  // No obstacles found
    }

    std::vector<double> TankEnv::reset()
    {
        controller_->resetImmediate();
        // initialize reward shaping memory
        auto objsPtr = controller_->getObjects();
        auto& objs = *objsPtr;
        if(objs.find(agent_tank_id_) != objs.end() && objs.find(enemy_tank_id_) != objs.end())
        {
            auto* me = dynamic_cast<Tank*>(objs[agent_tank_id_].get());
            auto* enemy = dynamic_cast<Tank*>(objs[enemy_tank_id_].get());
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
            if(diff > 180.0) diff = 360.0 - diff;
            prev_align_norm_ = diff / 180.0;
        }
        return getCurrentState();
    }

    std::tuple<std::vector<double>, double, bool> TankEnv::step(int action)
    {
        applyActionToAgent(action);
        // advance a fixed number of ticks to simulate one step
        for(int i = 0; i < 5; i++) controller_->stepOnce();
        step_counter_++;  // Increment step counter for timing
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
            // terminal: return zero vector (9 base + 64 map grid + 1 line-of-sight + 48 ray = 122)
            return std::vector<double>(9 + 64 + 1 + 16 * 3, 0.0);
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

        // ==================== 新增：全局地图信息 ====================
        // 添加地图块的简化表示，帮助agent理解整体布局
        // 使用网格表示法：将地图划分为 8x8 = 64个格子
        // 每个格子：1.0 表示有墙，0.0 表示空地
        const int MAP_GRID_SIZE = 8;
        const double CELL_WIDTH = GAME_VIEW_WIDTH / MAP_GRID_SIZE;
        const double CELL_HEIGHT = GAME_VIEW_HEIGHT / MAP_GRID_SIZE;
        
        auto* blocks = controller_->getBlocks();
        std::vector<double> map_grid(MAP_GRID_SIZE * MAP_GRID_SIZE, 0.0);
        
        // 标记所有有墙的格子
        for (const auto& kv : *blocks)
        {
            const Block& b = kv.second;
            auto bc = b.center();
            double bw = b.width();
            double bh = b.height();
            
            // 找出这个block覆盖的所有格子
            int min_gx = std::max(0, static_cast<int>((bc.x() - bw/2) / CELL_WIDTH));
            int max_gx = std::min(static_cast<int>(MAP_GRID_SIZE-1), static_cast<int>((bc.x() + bw/2) / CELL_WIDTH));
            int min_gy = std::max(0, static_cast<int>((bc.y() - bh/2) / CELL_HEIGHT));
            int max_gy = std::min(static_cast<int>(MAP_GRID_SIZE-1), static_cast<int>((bc.y() + bh/2) / CELL_HEIGHT));
            
            for (int gy = min_gy; gy <= max_gy; gy++)
            {
                for (int gx = min_gx; gx <= max_gx; gx++)
                {
                    map_grid[gy * MAP_GRID_SIZE + gx] = 1.0;  // 标记为有墙
                }
            }
        }
        
        // 添加到状态向量
        state.insert(state.end(), map_grid.begin(), map_grid.end());
        
        // ==================== 新增：直线视线标志 ====================
        // 添加一个特征：是否有到敌人的直线视线（0.0 或 1.0）
        state.push_back(hasDirectLineToEnemy(my.pos, en.pos) ? 1.0 : 0.0);

        std::vector<double> rays = rayFeatures();
        state.insert(state.end(), rays.begin(), rays.end());
        
        // 新状态维度：9 (基础) + 64 (地图网格) + 1 (直线视线) + 48 (射线) = 122
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

        // =================================================================
        // 1. 终端奖励 (Terminal Rewards) - 保持不变
        // 这是最强的信号，明确定义了最终目标。
        // =================================================================
        if (!meAlive || !enemyAlive)
        {
            done = true;
            if (meAlive && !enemyAlive) return 100.0;  // 胜利！获得巨大奖励
            if (!meAlive && enemyAlive) return -100.0; // 失败！受到巨大惩罚
            return 50.0;  // 平局 (双方同时被消灭)
        }

        // 初始化本轮的奖励
        double r = 0.0;

        // 获取智能体和敌人的状态
        auto* me = dynamic_cast<Tank*>(objs[agent_tank_id_].get());
        auto* enemy = dynamic_cast<Tank*>(objs[enemy_tank_id_].get());
        auto my = me->getCurrentPosition();
        auto en = enemy->getCurrentPosition();

        // =================================================================
        // 2. 步进惩罚 / 时间成本 (Step Penalty) - 优化
        // 鼓励效率，避免无限期的僵持。
        // 原来的 -0.0005 太小可以忽略不计，这里我们将其增大。
        // =================================================================
        r -= 0.01;

        // =================================================================
        // 3. 生存相关的奖惩 (Survival Rewards) - 关键优化
        // 这是防止智能体自杀行为的核心。
        // =================================================================

        // 3.1 撞墙/停滞惩罚 (Stagnation Penalty) - 新增！
        double move_dist = std::hypot(my.pos.x() - last_my_pos_.x(), my.pos.y() - last_my_pos_.y());
        if (move_dist < 0.5) // 如果坦克几乎没有移动，很可能被墙卡住了
        {
            r -= 0.5; // 给予一个显著的惩罚，告诉它“不要卡住不动或撞墙”
        }

        // 3.2 子弹规避奖惩 (Bullet Avoidance) - 优化
        // 大幅增加权重，使躲避子弹成为高优先级行为。
        double closest_bullet_dist = 1000.0;
        for (auto& kv : objs)
        {
            if (kv.second->type() == OBJ_SHELL)
            {
                Shell* sh = dynamic_cast<Shell*>(kv.second.get());
                auto sh_pos = sh->getCurrentPosition();
                double d = std::hypot(sh_pos.pos.x() - my.pos.x(), sh_pos.pos.y() - my.pos.y());
                closest_bullet_dist = std::min(closest_bullet_dist, d);
            }
        }

        if (closest_bullet_dist < prev_closest_bullet_dist_ && closest_bullet_dist < 150.0) // 增大探测范围
        {
            r -= 0.8;  // 靠近子弹是非常危险的，给予严厉惩罚
        }
        else if (closest_bullet_dist > prev_closest_bullet_dist_ && prev_closest_bullet_dist_ < 150.0)
        {
            r += 4;  // 成功远离子弹是好的行为，给予奖励
        }
        prev_closest_bullet_dist_ = closest_bullet_dist;


        // =================================================================
        // 4. 进攻相关的奖惩 (Offensive Rewards) - 优化
        // 鼓励有效的进攻行为。
        // =================================================================

        // 4.1 射击奖励 (Shooting Reward) - 增强版
        int current_shells = me->remainShells();
        if (current_shells < last_my_shells_)
        {
            double current_time = step_counter_ * 0.05;  // Each step ~50ms  
            recent_shot_times_.push_back(current_time);
            
            // 计算开火时的瞄准程度
            double bearing_fire = std::atan2(en.pos.y() - my.pos.y(), en.pos.x() - my.pos.x()) * 180.0 / M_PI;
            double diff_fire = std::fmod(std::fabs(bearing_fire - my.angle), 360.0);
            if (diff_fire > 180.0) diff_fire = 360.0 - diff_fire;
            double align_norm_fire = diff_fire / 180.0;

            // ==================== 新增：直线射击大奖励 ====================
            // 如果炮口对准敌人且之间无障碍物，给予巨大奖励（10秒冷却）
            if (align_norm_fire < 0.1)  // 瞄准精确（18度内）
            {
                if (hasDirectLineToEnemy(my.pos, en.pos))  // 检查是否有直线视线
                {
                    // 检查冷却时间（10秒）
                    if (current_time - last_direct_shot_time_ >= 10.0)
                    {
                        r += 5.0;  // 🎯 直线射击大奖励！
                        last_direct_shot_time_ = current_time;
                        // Note: This reward should encourage strategic positioning
                    }
                    else
                    {
                        r += 2.0;  // 冷却中，仍然给予普通精确射击奖励
                    }
                }
                else
                {
                    r += 0.1;  // 瞄准准确但有墙阻挡，普通奖励
                }
            }
            else if (align_norm_fire < 0.3) // ~54度以内，还行
            {
                r += 0.5;  // 奖励有价值的尝试
            }
            else
            {
                r -= 0.2;  // 惩罚浪费弹药（略微增加惩罚）
            }
        }
        last_my_shells_ = current_shells;
        
        // ==================== 新增：频繁射击惩罚 ====================
        // 清理超过3秒的旧射击记录
        double current_time = step_counter_ * 0.05;
        while (!recent_shot_times_.empty() && current_time - recent_shot_times_.front() > 3.0)
        {
            recent_shot_times_.pop_front();
        }
        
        // 如果3秒内射击超过3次，给予惩罚
        if (recent_shot_times_.size() > 3)
        {
            r -= 10;  // 🚫 频繁射击惩罚！避免无脑spam
        }


        // =================================================================
        // 5. 奖励塑形 / 战术引导 (Reward Shaping) - 优化
        // 这些是“微调”行为的奖励，作为次要目标引导智能体。
        // =================================================================
        double maxd = std::hypot(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT);
        double dist = std::hypot(en.pos.x() - my.pos.x(), en.pos.y() - my.pos.y());
        double dist_norm = dist / maxd;

        double bearing = std::atan2(en.pos.y() - my.pos.y(), en.pos.x() - my.pos.x()) * 180.0 / M_PI;
        double diff = std::fmod(std::fabs(bearing - my.angle), 360.0);
        if (diff > 180.0) diff = 360.0 - diff;
        double align_norm = diff / 180.0;

        // 5.1 靠近敌人奖励 (Getting Closer) - 关键优化
        // **权重被大幅降低**，现在它只是一个微小的激励，而不是驱动自杀行为的主要原因。
        r += 0.1 * (prev_dist_norm_ - dist_norm);

        // 5.2 瞄准敌人奖励 (Alignment)
        // 权重略微提高，使其比“靠近”更重要，鼓励智能体先瞄准再行动。
        r += 0.2 * (prev_align_norm_ - align_norm);

        // 5.3 原地旋转惩罚 (Spinning Penalty)
        double angle_change = std::fmod(std::fabs(my.angle - last_my_angle_), 360.0);
        if (angle_change > 180.0) angle_change = 360.0 - angle_change;
        if (move_dist < 1.0 && angle_change > 5.0)
        {
            r -= 0.05; // 略微增加惩罚，减少无意义的旋转
        }

        // =================================================================
        // 更新记忆，为下一步计算做准备
        // =================================================================
        last_my_pos_ = my.pos;
        last_my_angle_ = my.angle;
        last_enemy_pos_ = en.pos;
        prev_dist_norm_ = dist_norm;
        prev_align_norm_ = align_norm;

        return r;
    }
}


