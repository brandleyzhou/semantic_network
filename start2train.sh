#python train.py --model MONO --dataset cityscapes --use_focal  --classes 19 --max_epochs 350 --optim adam --lr_schedule poly  
#python train.py --feature_fusing True --model ENet --dataset cityscapes --use_focal  --classes 19 --max_epochs 350 --optim adam --lr_schedule poly  
python train.py --model FPENet --dataset cityscapes --use_focal  --classes 19 --max_epochs 350 --optim adam --lr_schedule poly  
