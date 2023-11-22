#!/usr/bin/env bash
# cd eval/
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval.py --config $2 --eval_scenes horns   
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval.py --config $2 --eval_scenes trex    
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval.py --config $2 --eval_scenes room    
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval.py --config $2 --eval_scenes flower  
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval.py --config $2 --eval_scenes orchids 
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval.py --config $2 --eval_scenes leaves  
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval.py --config $2 --eval_scenes fern    
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 python eval.py --config $2 --eval_scenes fortress

# CUDA_VISIBLE_DEVICES=$1 python eval.py --config ../configs/eval_llff.txt --eval_scenes horns   
