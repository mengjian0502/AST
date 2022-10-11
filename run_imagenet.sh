PYTHON="/home/YOURHOME/anaconda3/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./data
fi

export CUDA_VISIBLE_DEVICES=0,1,2

model=uresnet50_imagenet

# prune
prune_rate=0.5
init_density=1.0
final_density=0.05
update_frequency=4000

lr=0.1
wd=1e-4
epochs=150
dataset=imagenet
pruner=iterative
gmomentum=0.0
offset=400

data_path="YOUR_DATA_PATH"
batch_size=256

# multiple runs
lb=1
diff=1
ub=1

for i in $(seq ${lb} ${diff} ${ub})
do
save_path="./save/${dataset}/${pruner}/${model}_init${init_density}_0.1tofinal${final_density}_pr${prune_rate}_2subnets_update${update_frequency}_offset${offset}x2_epochs${epochs}/run${i}/"
log_file="${model}_lr${lr}_wd${wd}_train.log"
name="${pruner}_${model}_init${init_density}_0.5tofinal${final_density}_pr${prune_rate}_2subnets_update${update_frequency}_cos_lr_offset${offset}x2_gm${gmomentum}_run${i}"

$PYTHON -W ignore main.py \
    --model ${model} \
    --iter \
    --pc_grad \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --wd ${wd} \
    --epochs ${epochs} \
    --prune-rate ${prune_rate} \
    --init-density ${init_density} \
    --final-density ${final_density} \
    --update-frequency ${update_frequency} \
    --log_file ${log_file} \
    --save_path ${save_path} \
    --init-prune-epoch 5 \
    --final-prune-epoch 30 \
    --slist 0.5 0.05 \
    --ngpu 3 \
    --pruner ${pruner} \
    --label_smoothing 0.1 \
    --lr_scheduler step \
    --gmomentum ${gmomentum} \
    --wandb False \
    --name ${name} \
    --entity jmeng15 \
    --iteroffset ${offset} \
    --project neurips22;
done
