#!/bin/bash
echo "Activating huggingface environment"
source /share/apps/anaconda3/2021.05/bin/activate huggingface
echo "Beginning script"
cd /share/luxlab/andrea/religion-subreddits
python3 project-UMAP.py -n 100 -d 0.5
