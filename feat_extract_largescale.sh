# -------- extract features from imagenet-r50/mnet 
# ---- pre-trained (r50,mnet) and supervised contrastive learning (r50)
for out_data in iNaturalist SUN Places Texture
do

CUDA_VISIBLE_DEVICES=0 python feat_extract_largescale.py \
 --arch R50 --in_data ImageNet --out_data ${out_data} --batch_size 128

CUDA_VISIBLE_DEVICES=0 python feat_extract_largescale.py \
 --arch R50 --in_data ImageNet --out_data ${out_data} --batch_size 128 \
 --model_path ./save/ImageNet/R50/supcon/supcon-linear.pth --supcon

CUDA_VISIBLE_DEVICES=0 python feat_extract_largescale.py \
 --arch MNet --in_data ImageNet --out_data ${out_data} --batch_size 128

CUDA_VISIBLE_DEVICES=0 python feat_extract_largescale.py \
 --arch ViT --in_data ImageNet --out_data ${out_data} --batch_size 128

done
