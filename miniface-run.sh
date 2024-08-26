#/!bin/sh
cd mediapipe-vt-master
. bin/activate
python tracker.py --camera $1 --listen-killsig
