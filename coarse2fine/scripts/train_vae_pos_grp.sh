### bash /liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/scripts/train_vae_pos_grp.sh

# env
source /liujinxin/zhefei/miniforge3/bin/activate
conda activate roboar4il

# folder
cd /liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based

# show the number of cpu
lscpu | grep '^CPU(s):'
cat /proc/cpuinfo| grep "cpu cores"| uniq

# run-x
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
--bs=256 \
--ep=400 \
--data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
--model_name='bowl-cos-x' \
--exp_name='vq' \
--seed=42 \
--vocab_size=512 \
--act_dim_sep=0 \
--vqnorm=True \
--saving_interval=50 \
--workers=16 \

# run-y
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
--bs=256 \
--ep=400 \
--data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
--model_name='bowl-cos-y' \
--exp_name='vq' \
--seed=42 \
--vocab_size=512 \
--act_dim_sep=1 \
--vqnorm=True \
--saving_interval=50 \
--workers=16 \

# # run-z
# export OMP_NUM_THREADS=16
# OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
# --bs=256 \
# --ep=400 \
# --data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
# --model_name='bowl-cos-z' \
# --exp_name='vq' \
# --seed=42 \
# --vocab_size=512 \
# --act_dim_sep=2 \
# --vqnorm=True \
# --saving_interval=50 \
# --workers=16 \

# # run-gripper
# export OMP_NUM_THREADS=16
# OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
# --bs=256 \
# --ep=400 \
# --data_path='/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl.hdf5' \
# --model_name='bowl-cos-gripper' \
# --exp_name='vq' \
# --seed=42 \
# --vocab_size=512 \
# --act_dim_sep=9 \
# --vqnorm=True \
# --saving_interval=50 \
# --workers=16 \
