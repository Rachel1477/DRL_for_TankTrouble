//
// Created by zr on 23-3-9.
//
#include "Controller.h"

namespace TankTrouble
{
    Controller::Controller():
        started(false),
        snapshot(new ObjectList) {}

    Controller::ObjectListPtr Controller::getObjects()
    {
        std::lock_guard<std::mutex> lg(mu);
        return snapshot;
    }

    Controller::BlockList* Controller::getBlocks()
    {
        std::lock_guard<std::mutex> lg(blocksMu);
        return &blocks;
    }

    std::vector<PlayerInfo> Controller::getPlaysInfo()
    {
        std::vector<PlayerInfo> info;
        std::lock_guard<std::mutex> lg(playersInfoMu);
        for(const auto& entry: playersInfo)
            info.push_back(entry.second);
        return std::move(info);
    }

    Controller::~Controller()
    {
        if(controlThread.joinable())
            controlThread.join();
    }
}