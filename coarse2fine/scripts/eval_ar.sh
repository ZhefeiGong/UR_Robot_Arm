### bash /liujinxin/zhefei/ARGen4IL/workspace/realworld_image_based/scripts/eval_ar.sh

# download
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

####### 🫙 Can 🫙
python eval_ar.py \
--output_dir /liujinxin/zhefei/ARGen4IL/workspace/dpbench_image_based/results/can/act_ar_can-img-ly16-b64g1-im1-em160-ema_v512_103110/1300epo_ema \
--var_ckpt /liujinxin/zhefei/ARGen4IL/workspace/dpbench_image_based/local_output/act_ar_can-img-ly16-b64g1-im1-em160-ema_v512_103110/ar-ep_1300-te_0.98-tr_1.00.pth \
--dataset_path /liujinxin/zhefei/ARGen4IL/workspace/readworld_image_based/data/bowl.hdf5 \
--nobs 1 \
--nactions 8 \
--max_steps 400 \

# ####### 🪵 Square 🪵
# python eval.py \
# -o /liujinxin/zhefei/ARGen4IL/workspace/code_act_ar_xyz_r6d_g_sep/results/square/square_2obs_16pred_8act_cos_embed8-layer8-bc400_3000epo_top1_102019 \
# -r /liujinxin/zhefei/ARGen4IL/workspace/code_act_ar_xyz_r6d_g_sep/ckpt/ar/square/act_ar_square-act-cos-embed8-layer8-bc400_v512_102019/ar-ckpt-3000.pth \
# -d /liujinxin/zhefei/ARGen4IL/workspace/data/robomimic/datasets/square/ph/image_abs.hdf5 \
# -a 8 \
# -s 400

# ####### 🔧 Tool Hang 🔧
# python eval.py \
# -o /liujinxin/zhefei/ARGen4IL/workspace/code_act_ar_xyz_r6d_g_sep/results/square/square_2obs_16pred_8act_cos_embed8-layer8-bc400_2000epo_top1_102019 \
# -r /liujinxin/zhefei/ARGen4IL/workspace/code_act_ar_xyz_r6d_g_sep/ckpt/ar/square/act_ar_square-act-cos-embed8-layer8-bc400_v512_102019/ar-ckpt-2000.pth \
# -d "/liujinxin/zhefei/ARGen4IL/workspace/data/robomimic/datasets/tool_hang/ph/image_abs.hdf5" \
# -a 8 \
# -s 400\

# from torchvision.utils import save_image
# save_image(tensor_image, 'image.png')

# ####### REPO
# command="python eval.py \
# -o /liujinxin/zhefei/ARGen4IL/workspace/code_act_ar_xyz_r6d_g_sep_img/results/can/102317/can_2obs_16pred_8act_cos_em160-ly16-b128-im1_650epo_top1_102317 \
# -r /liujinxin/zhefei/ARGen4IL/workspace/code_act_ar_xyz_r6d_g_sep_img/local_output/act_ar_can-img-ly16-b128g2-im1-em160_v512_102317/ar-ckpt-650.pth \
# -d /liujinxin/zhefei/ARGen4IL/workspace/data/robomimic/datasets/can/ph/image_abs.hdf5 \
# -a 8 \
# -s 400"
# for i in $(seq 1050 50 1100); do
#     modified_command="${command//650/$i}"
#     echo "Executing: $modified_command"
#     eval "$modified_command"
# done


