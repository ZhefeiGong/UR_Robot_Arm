### bash /liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/scripts/train_ar_bowl2.sh

# run
apt update
apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# env
source /liujinxin/zhefei/miniforge3/bin/activate
conda activate roboar4il

# folder
cd /liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based

# show the number of cpu
lscpu | grep '^CPU(s):'
cat /proc/cpuinfo| grep "cpu cores"| uniq

# run
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_ar.py \
--bs=64 \
--ep=16000 \
--data_path="/liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/data/bowl/bowl.hdf5" \
--model_name='bowl-img-ly20-b64g1-im1-em160' \
--exp_name='ar' \
--task_name='bowl' \
--tdepth=20 \
--tembed=160 \
--tnobs=1 \
--seed=42 \
--vocab_size=512 \
--workers=16 \
--vqnorm=True \
--saving_interval=200 \
