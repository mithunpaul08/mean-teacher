#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=2:ncpus=16:mem=64gb:ngpus=1


### Specify a name for the job
#PBS -N job_name
### Specify the group name
#PBS -W group_list=group_name
### Used if job requires partial node only
#PBS -l place=pack:shared
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=1344:00:00
### Walltime is how long your job will run
#PBS -l walltime=48:00:00

#PBS -e /extra/mithunpaul/error/
#PBS -o /extra/mithunpaul/output/

#####module load cuda80/neuralnet/6/6.0
#####module load cuda80/toolkit/8.0.61
module load singularity/2/2.6.1

echo $PWD
date
cd ~/mean_teacher/student_teacher
date
echo $PWD

pip install numpy scipy pandas nltk tqdm sklearn comet_ml gitpython
pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

bash get_glove.sh
bash get_data_run.sh
