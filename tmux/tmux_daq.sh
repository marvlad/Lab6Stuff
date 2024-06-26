#!/bin/bash

# Start a new Tmux session named "toolchain"
tmux new-session -d -s toolchain

tmux set mouse on

# This part splits the panes
# -------------------------------------------------------
# Split the window vertically with -v and horizontally -h
tmux split-window -h

# Split the first window vertically
tmux split-window -v

# Switch to the pane 1
tmux select-pane -t 1

# Split the first window vertically
tmux split-window -v

# Switch to the pane 1
tmux select-pane -t 2

# Split the first window horizontally
tmux split-window -h

# Switch to the pane 1
tmux select-pane -t 2

# Split the first window vertically
tmux split-window -v


# This part send the commands to the panes
# -------------------------------------------------------

tmux send-keys -t toolchain:0.0 'cd /home/pi/ToolDAQ_LAPPD_SaveModeV3.0/ToolDAQ_LAPPD_SaveMode/' C-m 
tmux send-keys -t toolchain:0.1 'cd /home/pi/ToolDAQ_LAPPD_SaveModeV3.0/ToolDAQ_LAPPD_SaveMode/Results; watch -n 0.2 "ls -lrt | tail -n 4"' C-m 
tmux send-keys -t toolchain:0.2 'cd /home/pi/ToolDAQ_LAPPD_SaveModeV3.0/ToolDAQ_LAPPD_SaveMode/Results; watch -n 1 "find *txt| wc -l"' C-m 
tmux send-keys -t toolchain:0.3 'watch -n 30 "df --output=pcent /"' C-m 
tmux send-keys -t toolchain:0.4 'cd /home/pi/ToolDAQ_LAPPD_SaveModeV3.0/ToolDAQ_LAPPD_SaveMode/Results; watch -n 2 "head -12 ../configfiles/ReadOutChain/BoardsConfig | tail +12"' C-m 
tmux send-keys -t toolchain:0.5 'cd /home/pi/ToolDAQ_LAPPD_SaveModeV3.0/ToolDAQ_LAPPD_SaveMode/Results' C-m 

tmux attach -t toolchain
