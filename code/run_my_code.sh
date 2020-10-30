export PYTHONPATH=$PWD:$PYTHONPATH
python train.py --batch 4 --batch_vald 2 --lr 1e-3 --epochs 100
