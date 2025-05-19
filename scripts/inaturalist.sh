# Train ResNet
python inat_train_sam.py --gpu 0,1,2,3 --dataset inaturalist --seed 0 --loss_type LA --train_rule None --SAM_type Focal-SAM --flat_gamma 2.0 --sharpness 1.1 --rho 0.2 --cos_lr --rho_schedule step --rho_steps 0.2 0.2 --save_freq 200 --wd 2e-4 --root_log "./log/ResNet/inaturalist" --root_model "./log/ResNet/inaturalist" 


# Fine-tune CLIP
python inat_train_sam_clip.py --gpu 0 --dataset inaturalist --seed 0 --arch CLIP-ViT-B/16 --loss_type LA --train_rule None --SAM_type Focal-SAM --flat_gamma 0.2 --sharpness 1.0 --rho 0.02 --cos_lr --rho_schedule none --save_freq 20 --lr 0.01 --wd 5e-4 --root_log "./log/CLIP/inaturalist" --root_model "./log/CLIP/inaturalist" --prec fp32 --adaptformer --epochs 20 -b 128

