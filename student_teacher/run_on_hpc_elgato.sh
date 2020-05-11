#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=16:mem=62gb:pcmem=4gb
### Specify a name for the job
#PBS -N elgato_
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=672:00:00
### Walltime is how long your job will run
#PBS -l walltime=24:00:00
### Joins standard error and standard out
#PBS -j oe


#####module load cuda80/neuralnet/6/6.0
#####module load cuda80/toolkit/8.0.61
module load singularity/3.2.1

echo $PWD
date
cd ~/mean-teacher/student_teacher
date
echo $PWD


pip install numpy scipy pandas nltk tqdm sklearn comet_ml gitpython
conda install ninja pyyaml mkl mkl-include setuptools cmake cffi
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html


bash get_glove.sh
bash get_data_run.sh
