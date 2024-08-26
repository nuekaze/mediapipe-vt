@echo off
cd mediapipe-vt-master
call Scripts\activate
python tracker.py --camera $1 --listen-killsig
