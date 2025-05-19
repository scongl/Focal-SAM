# Train ResNet
python cifar_train_sam.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LA --train_rule None --gamma 0.05 --tau 0.75 --SAM_type Focal-SAM --rho 0.2 --rho_schedule none --dataset cifar10 --seed 0 --epochs 200 --save_freq 200 --cos_lr --flat_gamma 3.0 --sharpness 0.5 --wd 2e-4 -b 64 --root_log "log/ResNet/cifar10" --root_model "log/ResNet/cifar10"

python cifar_train_sam.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LA --train_rule None --gamma 0.05 --tau 0.75 --SAM_type Focal-SAM --rho 0.3 --rho_schedule none --dataset cifar100 --seed 0 --epochs 200 --save_freq 200 --cos_lr --flat_gamma 3.2 --sharpness 0.8 --wd 1e-3 -b 64 --root_log "log/ResNet/cifar100" --root_model "log/ResNet/cifar100"

# Fine-tune CLIP
python cifar_train_sam_clip.py --gpu 0 --imb_factor 0.01 --loss_type LA --SAM_type Focal-SAM --rho 0.05 --dataset cifar10 --seed 0 --flat_gamma 1.5 --sharpness 0.7 --arch CLIP-ViT-B/16 --root_log "./log/CLIP/cifar10" --root_model "./log/CLIP/cifar10" --epochs 20 --adaptformer --lr 0.01 --wd 5e-4

python cifar_train_sam_clip.py --gpu 0 --imb_factor 0.01 --loss_type LA --SAM_type Focal-SAM --rho 0.05 --dataset cifar100 --seed 0 --flat_gamma 5.5 --sharpness 0.8 --arch CLIP-ViT-B/16 --root_log "./log/CLIP/cifar100" --root_model "./log/CLIP/cifar100" --epochs 20 --adaptformer --lr 0.01 --wd 5e-4




