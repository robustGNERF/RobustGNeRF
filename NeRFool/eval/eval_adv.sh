OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval_adv.py --config $2 \
                                                            --adv_iters 1000 \
                                                            --adv_lr 1 \
                                                            --epsilon 8 \
                                                            --use_adam \
                                                            --adam_lr 1e-3 \
                                                            --lr_gamma=1 --view_specific \
                                                            --eval_scenes horns \
                                                            --export_adv_source_img

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval_adv.py --config $2 \
                                                            --adv_iters 1000 \
                                                            --adv_lr 1 \
                                                            --epsilon 8 \
                                                            --use_adam \
                                                            --adam_lr 1e-3 \
                                                            --lr_gamma=1 --view_specific \
                                                            --eval_scenes trex \
                                                            --export_adv_source_img

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval_adv.py --config $2 \
                                                            --adv_iters 1000 \
                                                            --adv_lr 1 \
                                                            --epsilon 8 \
                                                            --use_adam \
                                                            --adam_lr 1e-3 \
                                                            --lr_gamma=1 --view_specific \
                                                            --eval_scenes room \
                                                            --export_adv_source_img

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval_adv.py --config $2 \
                                                            --adv_iters 1000 \
                                                            --adv_lr 1 \
                                                            --epsilon 8 \
                                                            --use_adam \
                                                            --adam_lr 1e-3 \
                                                            --lr_gamma=1 --view_specific \
                                                            --eval_scenes flower \
                                                            --export_adv_source_img

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval_adv.py --config $2 \
                                                            --adv_iters 1000 \
                                                            --adv_lr 1 \
                                                            --epsilon 8 \
                                                            --use_adam \
                                                            --adam_lr 1e-3 \
                                                            --lr_gamma=1 --view_specific \
                                                            --eval_scenes orchids \
                                                            --export_adv_source_img

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval_adv.py --config $2 \
                                                            --adv_iters 1000 \
                                                            --adv_lr 1 \
                                                            --epsilon 8 \
                                                            --use_adam \
                                                            --adam_lr 1e-3 \
                                                            --lr_gamma=1 --view_specific \
                                                            --eval_scenes leaves \
                                                            --export_adv_source_img

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval_adv.py --config $2 \
                                                            --adv_iters 1000 \
                                                            --adv_lr 1 \
                                                            --epsilon 8 \
                                                            --use_adam \
                                                            --adam_lr 1e-3 \
                                                            --lr_gamma=1 --view_specific \
                                                            --eval_scenes fern \
                                                            --export_adv_source_img

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval_adv.py --config $2 \
                                                            --adv_iters 1000 \
                                                            --adv_lr 1 \
                                                            --epsilon 8 \
                                                            --use_adam \
                                                            --adam_lr 1e-3 \
                                                            --lr_gamma=1 --view_specific \
                                                            --eval_scenes fortress \
                                                            --export_adv_source_img
