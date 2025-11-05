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

    private:
        void choose_local();

        Gtk::Image bg;
        Gtk::Button localBtn;
        sigc::signal<void> choose_local_s;
    };
}

#endif //TANK_TROUBLE_ENTRY_VIEW_H
