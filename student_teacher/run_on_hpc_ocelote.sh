#!/bin/bash
#PBS -q windfall
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1:os7=True
#PBS -W group_list=msurdeanu
#PBS -l walltime=24:00:00
#PBS -j oe


cd /home/u11/mithunpaul/
module load cuda90/neuralnet/7/7.3.1.20
module load python/3.6/3.6.5

#uncomment this if you don't want to reinstall venv- usually you just have to do this only once ever
#rm -rf my_virtual_env
#mkdir my_virtual_env
#python3 -m venv my_virtual_env

#this is the only line you need if you already have a virtual_env set up
source my_virtual_env/bin/activate


#pip install --upgrade pip
#pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
#pip install -r requirements.txt




#####my code part
export PYTHONPATH="/home/u11/mithunpaul/mean-teacher/"
cd /home/u11/mithunpaul/mean-teacher/student_teacher


#remove these two lines if you want to run on full data
#bash get_fact_verification_files.sh
#bash reduce_size_fact_verification_files.sh



bash get_data.sh
bash convert_allnli_format_gzip.sh
bash run_main.sh