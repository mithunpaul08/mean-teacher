#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=224gb:np100s=1:os7=True
### Specify a name for the job
#PBS -N mithuns_meanteacher
### Optional. Request email when job begins and ends - commented out in this case
### PBS -m bea
### Optional. Specify email address to use for notification - commented out in this case
### PBS -M mithunpaul@email.arizona.edu
### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=48:00:00
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
pip install torch torchvision


bash get_data_run.sh