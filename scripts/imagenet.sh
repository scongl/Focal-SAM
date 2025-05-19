# Train ResNet
python imagenet_train_sam.py --gpu 0,1 --dataset imagenet --seed 0 --loss_type LA --train_rule None --SAM_type Focal-SAM --flat_gamma 0.8 --sharpness 0.8 --rho 0.05 --cos_lr --rho_schedule step --rho_steps 0.1 0.2 --save_freq 200 --wd '5e-4' --root_log "./log/ResNet/imagenet" --root_model "./log/ResNet/imagenet"

# Fine-tune CLIP
python imagenet_train_sam_clip.py --gpu 0 --loss_type LA --train_rule None --SAM_type Focal-SAM --rho 0.05 --rho_schedule none --dataset imagenet --seed 0 --flat_gamma 0.5 --sharpness 0.8  --save_freq 20 --root_log "./log/CLIP/imagenet" --root_model "./log/CLIP/imagenet" --arch CLIP-ViT-B/16 --epochs 20 --adaptformer --cos_lr --lr 0.01 --wd 5e-4 -b 128




