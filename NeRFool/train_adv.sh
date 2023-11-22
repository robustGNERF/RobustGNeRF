OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=$1 python3 train.py --config configs/pretrain_dp_patchformer_adv.txt

# CUDA_VISIBLE_DEVICES=0 python eval_adv.py --config ../configs/eval_llff.txt \
#                                         --expname test --num_source_views 4 \
#                                         --adv_iters 1000 --adv_lr 1 \
#                                         --epsilon 8 --use_adam \
#                                         --adam_lr 1e-3 --lr_gamma=1 \
#                                         --view_specific
