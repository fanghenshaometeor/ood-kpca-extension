arch=R50
score=Energy
CUDA_VISIBLE_DEVICES=1 python energy_training.py \
 --arch ${arch} --score ${score} --in_data ImageNet --out_data iNaturalist

arch=R50
score=Energy
CUDA_VISIBLE_DEVICES=1 python energy_training.py \
 --arch ${arch} --supcon --model_path ./save/ImageNet/${arch}/supcon/supcon-linear.pth \
 --score ${score} --in_data ImageNet --out_data iNaturalist

arch=ViT
score=Energy
CUDA_VISIBLE_DEVICES=1 python energy_training.py \
 --arch ${arch} --score ${score} --in_data ImageNet --out_data iNaturalist

arch=R50
score=ReAct
CUDA_VISIBLE_DEVICES=0 python energy_training.py \
 --arch ${arch} --score ${score} --in_data ImageNet --out_data iNaturalist

arch=MNet
score=ReAct
CUDA_VISIBLE_DEVICES=1 python energy_training.py \
 --arch ${arch} --score ${score} --in_data ImageNet --out_data iNaturalist

