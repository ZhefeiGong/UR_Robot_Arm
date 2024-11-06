### bash /liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/scripts/train_vae_rot2.sh

# env
source /liujinxin/zhefei/miniforge3/bin/activate
conda activate roboar4il

# folder
cd /liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based

# show the number of cpu
lscpu | grep '^CPU(s):'
cat /proc/cpuinfo| grep "cpu cores"| uniq

# # run-r1
# export OMP_NUM_THREADS=16
# OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
# --bs=256 \
# --ep=400 \
# --data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
# --model_name='bowl-cos-r1' \
# --exp_name='vq' \
# --seed=42 \
# --vocab_size=512 \
# --act_dim_sep=3 \
# --vqnorm=True \
# --saving_interval=50 \
# --workers=16 \

# # run-r2
# export OMP_NUM_THREADS=16
# OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
# --bs=256 \
# --ep=400 \
# --data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
# --model_name='bowl-cos-r2' \
# --exp_name='vq' \
# --seed=42 \
# --vocab_size=512 \
# --act_dim_sep=4 \
# --vqnorm=True \
# --saving_interval=50 \
# --workers=16 \

# # run-r3
# export OMP_NUM_THREADS=16
# OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
# --bs=256 \
# --ep=400 \
# --data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
# --model_name='bowl-cos-r3' \
# --exp_name='vq' \
# --seed=42 \
# --vocab_size=512 \
# --act_dim_sep=5 \
# --vqnorm=True \
# --saving_interval=50 \
# --workers=16 \

# run-r4
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
--bs=256 \
--ep=400 \
--data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
--model_name='bowl-cos-r4' \
--exp_name='vq' \
--seed=42 \
--vocab_size=512 \
--act_dim_sep=6 \
--vqnorm=True \
--saving_interval=50 \
--workers=16 \

# run-r5
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
--bs=256 \
--ep=400 \
--data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
--model_name='bowl-cos-r5' \
--exp_name='vq' \
--seed=42 \
--vocab_size=512 \
--act_dim_sep=7 \
--vqnorm=True \
--saving_interval=50 \
--workers=16 \

# run-r6
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
--bs=256 \
--ep=400 \
--data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
--model_name='bowl-cos-r6' \
--exp_name='vq' \
--seed=42 \
--vocab_size=512 \
--act_dim_sep=8 \
--vqnorm=True \
--saving_interval=50 \
--workers=16 \
