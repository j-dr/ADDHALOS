#!/bin/bash
#SBATCH -p bigmem                          # partition (queue)
#SBATCH --qos bigmem                       # quality of service
#SBATCH -N 1                               # number of nodes
#SBATCH -n 32                              # cpu/node
#SBATCH --mem-per-cpu 48000                # MB per cpu
#SBATCH -t 0-16:00                         # run time
#SBATCH --get-user-env                     # export user environment vars
#SBATCH -o logs/train_C250_99.%N.%j.out       # STDOUT
#SBATCH -e logs/train_C250_99.%N.%j.err       # STDERR
#SBATCH --mail-type=END,FAIL               # notifications for job done & fail
#SBATCH --mail-user=joe.derose13@gmail.com # send-to address

python traintest.py /home/jderose/projects/addhalos/test/cfgs/addsnap_C250_99.cfg
