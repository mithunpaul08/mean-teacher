#!/bin/bash
#PBS -q standard
#PBS -l select=1:ncpus=16:mem=62gb:pcmem=4gb

### Specify a name for the job
#PBS -N elgato_
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=free:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=672:00:00
### Walltime is how long your job will run
#PBS -l walltime=24:00:00


### Optional. Request email when job begins and ends
# PBS -m bea
### Optional. Specify email address to use for notification
# PBS -M mithunpaul@email.arizona.edu

### Joins standard error and standard out
#PBS -j oe


module load singularity/3.2.1



echo $PWD
date
cd ~/mean-teacher/student_teacher
date
echo $PWD


pip install numpy scipy pandas nltk tqdm sklearn comet_ml gitpython
conda install ninja pyyaml mkl mkl-include setuptools cmake cffi
pip install torch torchvision

bash get_glove.sh
bash get_data_run.sh

