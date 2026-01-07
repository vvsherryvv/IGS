#!/bin/bash
#

videopath="$1"
# 新增：接收帧数参数，默认为 300
n_frames=${2:-300} 
last_frame_id=$((n_frames - 1))

# 传递参数给 python 脚本
python pre_input.py --videopath "$videopath" --endframe "$n_frames"
python my_convert.py -s "$videopath/colmap_0"
python my_copy_cams.py --source "$videopath/colmap_0" --scene "$videopath"
# 传递 last_frame_id 给 my_convert_frames
python my_convert_frames.py -s "$videopath" --last_frame_id "$last_frame_id"