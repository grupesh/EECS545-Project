export PYTHONPATH=$PWD:$PYTHONPATH
#rm -r '../saved_data'
#python3.8 gen_test_annotations.py
#python3.8 gen_validation_set.py
python train.py --batch 4 --batch_vald 2 --lr 1e-3 --epochs 100 --root_path 'G:/Shared drives/EECS 545 - ML Project/data'
