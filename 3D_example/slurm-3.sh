#!/bin/sh

#SBATCH --error=run.03.error
#SBATCH --output=run.03.out
#SBATCH --partition=allq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --job-name=tur_03
#SBATCH --mail-user=j.ruan@students.uu.nl
#SBATCH --mail-type=ALL
#SBATCH --export=all

# export ISHOT=$SLURM_ARRAY_TASK_ID 
# echo $ISHOT

source activate py3p7

my_file=beta3_nocross
echo $my_file

cd $SLURM_SUBMIT_DIR

if [ ! -d "error" ]; then
  mkdir error
fi

if [ ! -d "output" ]; then
  mkdir output
fi

# ------------ run script ------------------
python $my_file.py

echo "Running screipt $my_file.py"  >> log.txt
hostname >> log.txt
date >> log.txt
# ------------------------ end ---------------------------------------

echo "done screipt $my_file.py"  >> log.txt
hostname >> log.txt
date >> log.txt	

# ------------ move error and output files ------------------

# mv run.$my_file.error error/
# mv run.$my_file.out output/
