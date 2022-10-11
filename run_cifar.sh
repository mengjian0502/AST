PYTHON="/home/YOURHOME/anaconda3/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./data
fi

export CUDA_VISIBLE_DEVICES=0

model=uwresnet32

# prune
prune_rate=0.5
init_density=1
final_density=0.02
update_frequency=999

lr=0.1
wd=0.0005
epochs=160
dataset=cifar10
pruner=iterative
gmomentum=0.0
offset=100
reset_buffer=False


# multiple runs
lb=1
diff=1
ub=3

for i in $(seq ${lb} ${diff} ${ub})
do
save_path="./save/${dataset}/${pruner}/epochs${epochs}/${model}_init${init_density}_0.1tofinal${final_density}_pr${prune_rate}_1subnets_update${update_frequency}_offset${offset}x3/run${i}/"
log_file="${model}_lr${lr}_wd${wd}_train.log"
name="${pruner}_${dataset}_${model}_init${init_density}_0.1tofinal${final_density}_pr${prune_rate}_3subnets_update${update_frequency}_cos_lr_offset${offset}x3_gm${gmomentum}_run${i}"

$PYTHON -W ignore main.py \
    --model ${model} \
    --dataset ${dataset} \
    --lr ${lr} \
    --wd ${wd} \
    --epochs ${epochs} \
    --prune-rate ${prune_rate} \
    --init-density ${init_density} \
    --final-density ${final_density} \
    --update-frequency ${update_frequency} \
    --log_file ${log_file} \
    --save_path ${save_path} \
    --final-prune-epoch 110 \
    --slist 0.1 0.05 0.02 \
    --pruner ${pruner} \
    --lr_scheduler cosine \
    --gmomentum ${gmomentum} \
    --reset_buffer ${reset_buffer} \
    --wandb False \
    --name ${name} \
    --entity jmeng15 \
    --iter \
    --iteroffset ${offset} \
    --project neurips22_cifar100;
done
