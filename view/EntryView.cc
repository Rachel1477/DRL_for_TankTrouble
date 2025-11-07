//
// Created by zr on 23-3-3.
//

#include "EntryView.h"
#include "defs.h"

namespace TankTrouble
{
    EntryView::EntryView():
        bg("entry.jpg")
    {
        localBtn.set_label("单人游戏");
        localBtn.set_size_request(80, 60);
        localBtn.signal_clicked().connect(
                sigc::mem_fun(*this, &EntryView::choose_local));
        
        rlTrainingBtn.set_label("Agent训练");
        rlTrainingBtn.set_size_request(80, 60);
        rlTrainingBtn.signal_clicked().connect(
                sigc::mem_fun(*this, &EntryView::choose_rl_training));
        
        bg.set_size_request(WINDOW_WIDTH, WINDOW_HEIGHT);

        put(bg, 0, 0);
        put(localBtn, 170, 250);
        put(rlTrainingBtn, 170, 320);

        bg.show();
        localBtn.show();
        rlTrainingBtn.show();
    }

    void EntryView::choose_local() {choose_local_s.emit();}
    void EntryView::choose_rl_training() {choose_rl_training_s.emit();}

    sigc::signal<void> EntryView::signal_choose_local() {return choose_local_s;}
    sigc::signal<void> EntryView::signal_choose_rl_training() {return choose_rl_training_s;}
}