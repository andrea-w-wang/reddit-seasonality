#!/bin/bash
echo "Activating huggingface environment"
source /share/apps/anaconda3/2021.05/bin/activate huggingface
echo "Beginning script"
cd /share/luxlab/andrea/religion-subreddits
python3 calc_sample_perplexity.py -r Judaism -fp ./data/samples/Judaism-sample1676613006.pk -month 2014-01
#ST_DT='2014-01-01'
#EN_DT='2018-10-07'
#endt=$(date '+%s' -d "$EN_DT")
#i="$ST_DT"

#while [[ $(date +%s -d $i) -le $endt ]]; do
#   python3 calc_perplexity.py --subreddit Judaism -month ${i%-*}
#   i=$(date '+%Y-%m-%d' -d "$i +1 month")
#done

echo "=== Processing complete ==="


