//
// Created by zr on 23-2-16.
//

#ifndef TANK_TROUBLE_CONTROL_EVENT_H
#define TANK_TROUBLE_CONTROL_EVENT_H

namespace TankTrouble
{
    class ControlEvent
    {
    public:
        //前进，停止，后退，顺时针调头，逆时针调头
        enum Operation
        {
            Forward, Backward, RotateCW, RotateCCW, Fire,
            StopForward, StopBackward, StopRotateCW, StopRotateCCW
        };

        explicit ControlEvent(Operation op);
        ControlEvent();
        ~ControlEvent() = default;
        [[nodiscard]] Operation operation() const;

    private:
        Operation op;
    };
}

#endif //TANK_TROUBLE_CONTROL_EVENT_H
