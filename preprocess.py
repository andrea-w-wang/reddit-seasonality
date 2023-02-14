import re
import html
import sys
from unicodedata import category
punctuation_chars =  [chr(i) for i in range(sys.maxunicode) 
                             if category(chr(i)).startswith("P")]

punctuation_chars = ''.join(punctuation_chars)
def preprocess(text):
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    
    # remove relational link such as:  [contact the moderators of this subreddit](/message/compose/?to=/r/Buddhism) 
    opening_braces = '\(\/'
    closing_braces = '\)'
    non_greedy_wildcard = '.*?'
    text = re.sub(f'[{opening_braces}]{non_greedy_wildcard}[{closing_braces}]', '', text)
    
    # remove linked text such as "[^Exclude ^me]"
    opening_braces = '\[\^'
    closing_braces = '\]'
    non_greedy_wildcard = '.*?'
    text = re.sub(f'[{opening_braces}]{non_greedy_wildcard}[{closing_braces}]', '', text)
    
    # remove Bot message e.g. "^HelperBot ^v1.1 ^/r/HelperBot_ ^I ^am ^a ^bot. ^Please ^message ^/u/swim1929 ^with ^any ^feedback ^and/or ^hate. ^Counter: ^144517"
    text = re.sub(f'[\^]({non_greedy_wildcard})($|\s)', '', text)

    # remove punctuations
    text = text.translate(str.maketrans("", "", punctuation_chars+"><=^|$+~`"))
    text = re.sub("\n", " ", text)
    return text