# ======== ======== ======== ======== ====
# ======== R50/ViT, standard training ========
# ======== ======== ======== ======== ====
python main.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx NYS --gamma 0.15 --M 2048 --exp_var_ratio 0.995 # Nystrom, standard training

python main.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx RFF --gamma 3 --M 4096 --exp_var_ratio 0.5 # random Fourier features, standard training

python main.py \
 --arch ViT --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx NYS --gamma 0.2 --M 2048 --exp_var_ratio 0.96 # Nystrom, standard training

python main.py \
 --arch ViT --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx RFF --gamma 0.1 --M 2048 --exp_var_ratio 0.9 # Nystrom, standard training

# ======== ======== ======== ======== ======== ======
# ======== R50, supervised contrast learning ========
# ======== ======== ======== ======== ======== ======
python main.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx NYS --gamma 0.75 --M 1536 --exp_var_ratio 0.998 --supcon # Nystrom, supervised constrastive learning

python main.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx RFF --gamma 1 --M 4096 --exp_var_ratio 0.5 --supcon # random Fourier features, supervised constrastive learning

# ======== ======== ======== ======== ======== ======== ======== =======
# ======== R50/MNet, standard training, + feature rectification ========
# ======== ======== ======== ======== ======== ======== ======== =======
CUDA_VISIBLE_DEVICES=0 python main_fuse.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx NYS --gamma 0.6 --M 2048 --exp_var_ratio 0.99 # nystrom, standard training

CUDA_VISIBLE_DEVICES=0 python main_fuse.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx RFF --gamma 0.5 --M 4096 --exp_var_ratio 0.8 # random Fourier features, standard training

CUDA_VISIBLE_DEVICES=0 python main_fuse.py \
 --arch MNet --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx NYS --gamma 0.2 --M 2560 --exp_var_ratio 0.996 # nystrom, standard training

CUDA_VISIBLE_DEVICES=0 python main_fuse.py \
 --arch MNet --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --approx RFF --gamma 1 --M 2560 --exp_var_ratio 0.6 # random Fourier features, standard training