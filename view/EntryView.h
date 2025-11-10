//
// Created by zr on 23-3-3.
//

#ifndef TANK_TROUBLE_ENTRY_VIEW_H
#define TANK_TROUBLE_ENTRY_VIEW_H
#include <gtkmm.h>

namespace TankTrouble
{
    class EntryView : public Gtk::Fixed
    {
    public:
        EntryView();
        sigc::signal<void> signal_choose_local();
        sigc::signal<void> signal_choose_rl_training();

    private:
        void choose_local();
        void choose_rl_training();

        Gtk::Image bg;
        Gtk::Button localBtn;
        Gtk::Button rlTrainingBtn;
        sigc::signal<void> choose_local_s;
        sigc::signal<void> choose_rl_training_s;
    };
}

#endif //TANK_TROUBLE_ENTRY_VIEW_H
