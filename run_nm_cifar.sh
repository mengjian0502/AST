PYTHON="/home/YOURHOME/anaconda3/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./data
fi

export CUDA_VISIBLE_DEVICES=0

model=uresnet18

# prune
prune_rate=0.5
init_density=1
final_density=0.125
update_frequency=999

lr=0.1
wd=0.0005
epochs=200
pruner=nm_iterative
dataset=cifar10

save_path="./save/nm_sparsity/${dataset}/${model}_init${init_density}_final0.5to${final_density}_pr${prune_rate}_spars_grad/"
log_file="${model}_lr${lr}_wd${wd}_train.log"

$PYTHON -W ignore nm_main.py \
    --model ${model} \
    --dataset ${dataset} \
    --iter \
    --lr ${lr} \
    --wd ${wd} \
    --epochs ${epochs} \
    --prune-rate ${prune_rate} \
    --init-density ${init_density} \
    --final-density ${final_density} \
    --update-frequency ${update_frequency} \
    --log_file ${log_file} \
    --save_path ${save_path} \
    --final-prune-epoch 150 \
    --pruner ${pruner} \
    --pc_grad \
    --Mlist 4 4 8 \
    --Nlist 2 1 1;
