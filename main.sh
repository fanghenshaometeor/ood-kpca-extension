python main.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method NYS --gamma 0.15 --M 2048 --exp_var_ratio 0.995 # Nystrom, standard training

python main.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method RFF --gamma 3 --M 4096 --exp_var_ratio 0.5 # random Fourier features, standard training

python main.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method NYS --gamma 0.75 --M 1536 --exp_var_ratio 0.998 --supcon # Nystrom, supervised constrastive learning

python main.py \
 --arch R50 --in_data ImageNet --out_datasets iNaturalist SUN Places Texture \
 --method RFF --gamma 1 --M 4096 --exp_var_ratio 0.5 --supcon # random Fourier features, supervised constrastive learning