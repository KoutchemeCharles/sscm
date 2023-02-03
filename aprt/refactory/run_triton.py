import os 
from textwrap import dedent
from src.sscm.utils.files import create_dir

def create_bash_submit(log_folder, job_name, save_path, n_questions, percentage=100):
    
    create_dir(log_folder, clear=True)
    template = f"""
        #!/bin/bash -l
        #SBATCH --job-name={job_name}
        #SBATCH --time=08:00:00
        #SBATCH --cpus-per-task=2
        #SBATCH --mem=8GB
        #SBATCH --chdir=/home/koutchc1/refactory
        #SBATCH --output={log_folder}/slurm_%A_%a.out
        #SBATCH --array=1-{n_questions}

        module load miniconda;

        python3 run.py -d {save_path} -q question_$SLURM_ARRAY_TASK_ID -s {percentage} -o -m -b
        python3 run.py -d {save_path} -q question_$SLURM_ARRAY_TASK_ID -s {percentage} -f -m -b
    """
    # Read https://github.com/githubhuyang/refactory 
        
    return dedent(template).lstrip()
