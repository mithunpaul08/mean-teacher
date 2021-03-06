#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1:os7=True
### Specify a name for the job
#PBS -N 3t
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:shared
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours

#PBS -l cput=420:00:00
### Walltime is how long your job will run
#PBS -l walltime=15:00:00
### Joins standard error and standard out
#PBS -j oe


### Optional. Request email when job begins and ends
# PBS -m bea
### Optional. Specify email address to use for notification
# PBS -M mithunpaul@email.arizona.edu

module load cuda10
module load python/3.8
mkdir my_virtual_env
python3 -m venv my_virtual_env/
source my_virtual_env/bin/activate

echo $PWD
date
cd ~/mean-teacher/student_teacher
date
echo $PWD

pip install numpy scipy pandas nltk tqdm sklearn comet_ml gitpython
pip install torch torchvision


bash get_data_run.sh