log=$1
echo "cd /fs/clip-ml/he/opponent; \
    python scripts/read_protobowl_log.py --log $log --output_dir /fs/clip-scratch/hhe/qanta-buzz/parts --db /fs/clip-quiz/data/2015_12_14.db" | \
qsub -N qb-$log -q wide -l walltime=2:00:00,pmem=4g -o $log.out -e $log.err 
