//
// Created by zr on 23-2-16.
//

#ifndef TANK_TROUBLE_WINDOW_H
#define TANK_TROUBLE_WINDOW_H
#include <gtkmm.h>
#include <memory>
#include "view/EntryView.h"
#include "Controller.h"

namespace TankTrouble
{
    class GameView;
    

    class Window : public Gtk::Window
    {
    public:
        Window();
        ~Window() override;
        bool on_key_press_event(GdkEventKey* key_event) override;
        bool on_key_release_event(GdkEventKey* key_event) override;

        void notifyLoginSuccess();
        void notifyRoomUpdate();
        void notifyGameOn();
        void notifyGameOff();

    private:

        void onUserChooseLocal();
        void onUserChooseRLTraining();
        void toEntryView();

        void onLoginSuccess();
        void onRoomsUpdate();
        void onGameBegin();
        void onGameOff();

        std::unique_ptr<Controller> ctl;
        std::unique_ptr<GameView> gameView;
        EntryView entryView;

        Glib::Dispatcher loginSuccessNotifier;
        Glib::Dispatcher roomUpdateNotifier;
        Glib::Dispatcher gameOnNotifier;
        Glib::Dispatcher gameOffNotifier;

        bool KeyUpPressed, KeyDownPressed,
        KeyLeftPressed, KeyRightPressed,
        spacePressed;
    };
}

#endif //TANK_TROUBLE_WINDOW_H
