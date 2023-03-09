import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-py", "--py_func", required=True, type=str)
parser.add_argument("-s", "--subreddits", required=True, type=str, nargs="+")
parser.add_argument("-v", "--variable", type=str)
args = parser.parse_args()
variable_name = None
variable_value = None
if args.variable:
    variable_name = args.variable.split("=")[0]
    variable_value = args.variable.split("=")[1]

job_prefix = args.py_func.split(".")[0]
# write sh file
for subreddit in args.subreddits:
    with open(f"./{job_prefix}-{subreddit}.sh", "w") as f:
        if variable_value:
            f.write(
                f"""#!/bin/bash
    echo "Activating huggingface environment"
    source /share/apps/anaconda3/2021.05/bin/activate huggingface
    echo "Beginning script"
    cd /share/luxlab/andrea/religion-subreddits
    python3 {args.py_func} --subreddit {subreddit}
                """
            )
        else:
            f.write(
                f"""#!/bin/bash
                echo "Activating huggingface environment"
                source /share/apps/anaconda3/2021.05/bin/activate huggingface
                echo "Beginning script"
                cd /share/luxlab/andrea/religion-subreddits
                python3 {args.py_func} --subreddit {subreddit} --{variable_name} {variable_value}
                            """
        )

    with open(f"./{job_prefix}-{subreddit}.sub", "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH -J {job_prefix}-{subreddit}                            # Job name
#SBATCH -o /share/luxlab/andrea/religion-subreddits/logs/{job_prefix}-{subreddit}_%j.out # output file (%j expands to jobID)
#SBATCH -e /share/luxlab/andrea/religion-subreddits/logs/{job_prefix}-{subreddit}_%j.err # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                        # Request status by email
#SBATCH --mail-user=aww66@cornell.edu          # Email address to send results to.
#SBATCH -N 1                                   # Total number of nodes requested
#SBATCH -n 8                                  # Total number of cores requested
#SBATCH --get-user-env                         # retrieve the users login environment
#SBATCH --mem=50G                             # server memory requested (per node)
#SBATCH -t 5:00:00                            # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition          # Request partition
/share/luxlab/andrea/religion-subreddits/{job_prefix}-{subreddit}.sh
            """
        )
    os.system(f"chmod 775 {job_prefix}-{subreddit}.sh")
    os.system(f"sbatch --requeue {job_prefix}-{subreddit}.sub")