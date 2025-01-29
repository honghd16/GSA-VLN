DATA_ROOT=../datasets

train_alg=dagger

features=clip.b16
ft_dim=512
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=627

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-init.aug.45k

outdir=../datasets/R2R/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
    --dataset r2r
    --output_dir ${outdir}
    --world_size ${ngpus}
    --seed ${seed}
    --tokenizer bert      

    --enc_full_graph
    --graph_sprels
    --fusion dynamic

    --expert_policy spl
    --train_alg ${train_alg}
    
    --num_l_layers 9
    --num_x_layers 4
    --num_pano_layers 2
    
    --max_action_len 15
    --max_instr_len 200

    --batch_size 4
    --lr 1e-5
    --iters 200000
    --log_every 1000
    --optim adamW

    --features ${features}
    --image_feat_size ${ft_dim}
    --angle_feat_size 4

    --ml_weight 0.2   

    --feat_dropout 0.4
    --dropout 0.5
    
    --gamma 0."

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
    --tokenizer bert \
    --bert_ckpt_file ../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker/ckpts/pretrain_model_full_graph.pt \
    --aug ../datasets/R2R/annotations/ESA_Dataset/Train/prevalent_aug_train_enc.json \
    --max_traj_num 50 \
    --eval_first

# eval
MODEL_DIR="../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-init.aug.45k/ckpts"
for model in "$MODEL_DIR"/*; do
    echo "Evaluating model: $model"
    CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag \
        --tokenizer bert \
        --resume_file "$model" \
        --test
done
