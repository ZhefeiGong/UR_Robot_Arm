
# env
source /home/robot/miniforge3/bin/activate
conda activate roboar4il

# folder
cd /home/robot/UR_Robot_Arm/coarse2fine/

# show the number of cpu
lscpu | grep '^CPU(s):'
cat /proc/cpuinfo| grep "cpu cores"| uniq

