# Submits interactive job to the scheduler
salloc --time=0:30:00 --ntasks=2 --cpus-per-task=2 --mem-per-cpu=1G --gres=gpu:1  --nodes=2


# Then do
# module load python/3.10
# virtualenv -p python3.10 ~/cc_ssl