import os 
from textwrap import dedent
from src.utils.files import create_dir

def create_bash_submit(dataset_name, save_path, bash_save_path):
    log_folder = f"/home/koutchc1/dssc/logs/refactory_aprt/{dataset_name}"
    create_dir(log_folder, clear=True)
    template = f"""
        #!/bin/bash -l
        #SBATCH --job-name=refactory_repair_on_{dataset_name}_dataset
        #SBATCH --time=01:00:00
        #SBATCH --cpus-per-task=2
        #SBATCH --mem=4GB
        #SBATCH --chdir=/home/koutchc1/refactory
        #SBATCH --output={log_folder}/slurm_%A_%a.out
        #SBATCH --array=1-{len(os.listdir(save_path))}

        module load miniconda;

        python3 run.py -d {save_path} -q question_$SLURM_ARRAY_TASK_ID -s 100 -f -m -b
        python3 run.py -d {save_path} -q question_$SLURM_ARRAY_TASK_ID -s 100 -o -m -b
    """
    # Read https://github.com/githubhuyang/refactory 
        
    template = dedent(template).lstrip()
    with open(bash_save_path, "w") as fp:
        fp.write(template)