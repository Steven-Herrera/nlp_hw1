"""A module for completing Task 1 of Homework 1 for CSCI B651 Natural Language Processing

Task 1: Regex and building vocabulary of a corpus

TODO:
    [X] Build a regex pattern to detect all the headlines that mention all the pronoun names (Donald Trump,
       Joe Biden, Thai Le, Indiana University Bloomington) in the dataset. List all the headlines that you can
       find and print out how many unique names you can collect.

    [X] Build a set of vocabulary from all the headlines. This can be done through word tokenization, text
       normalization--e.g., case folding, lemmatization. Remember, a set of vocabulary contains only unique
       types of words. Please print out the size of your vocabulary.

    [] Count the frequency of each vocabulary across all the headlines. Plot a plot showing such frequency
       distribution for the top 100 most frequent words: the x-axis represents each word, and the y-axis
       represents the frequency of the corresponding word.
    """

from nltk.tokenize import word_tokenize
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import string

def _return_pattern():
    """Returns a generic regex pattern for finding names"""
    patterns = [
        "[A-aZ-z]+ [A-Z]\. [A-aZ-z]+", # matches names like Michael J. Fox
        "Mrs?\. [A-aZ-z]+", # Matches names like Mr. Smith and Mrs. Smith
        "Dr\. [A-aZ-z]+", # Matches names like Dr. Smith
        "Ms\. [A-aZ-z]+", # Matches names like Ms. Smith
        "Misses [A-aZ-z]+", # Matches names like Misses Smith
        "Pope [A-aZ-z]+", # Matches names like Pope Francis
        "Prince [A-aZ-z]+",
        "Princess [A-aZ-z]+",
        "Queen [A-aZ-z]+",
        "^[A-Z][a-z]+ [A-aZ-z]+ [A-Z][a-z]+(?=, .*, Dies at [0-9]+)", # "Kirk Peter Smith, famous person, Dies at 88"
        "^[A-Z][a-z]+ [A-Z][a-z]+(?=, .*, Dies at [0-9]+)", # "Kirk Smith, famous person, Dies at 88"
        "^[A-aZ-z]+ [A-aZ-z]+(?=, .*, Is Dead at [0-9]+)", # Kirk Smith, famous person, Is Dead at 88
        "^[A-aZ-z]+ [A-aZ-z]+ [A-aZ-z]+(?=, .*, Is Dead at [0-9]+)" # similar to above
        "^[A-Z][a-z]{4,} [A-Z][a-z]{3,}(?=’s [AEIOU])", # Aretha Franklin's ...
        "^[A-aZ-z]{6,} [A-aZ-z]{6,13}(?=’s)" # 
        "[A-aZ-z]+ [A-aZ-z]+(?=, [0-9]+, .*, Dies)",
        "[A-aZ-z]+ [A-aZ-z]+ [A-aZ-z]+(?=, [0-9]+, .*, Dies)"
        "[A-aZ-z]+ [A-aZ-z]+ Jr\.",
    ]
    pattern = "|".join(patterns)
    return pattern

def task1a(filepath = 'nytimes_data_final.csv'):
    pattern = _return_pattern()
    df = pd.read_csv(filepath)
    headlines = df['text'].tolist()
    n = len(headlines)
    results = []
    for i in range(n):
        hl = headlines[i]
        m = re.findall(pattern, hl)
        if len(m) > 0:
            print(hl)
            results.extend(m)
    # r = len(results)
    unique_names = set(results)
    u = len(unique_names)
    # print(f"Num Names: {r}")
    print(f"Unique Names: {u}")

def normalize(text):
    """Tokenizes a headline from the nytimes dataset"""
    punct = string.punctuation + '‘' + '’'
    for p in punct:
        text = text.replace(p, '')
    # text = text.replace('’', '')
    lowered_text = text.casefold().split()
    word_lst = list(set(lowered_text)) if len(lowered_text) > 1 else lowered_text
    new_text = " ".join(word_lst)
    return new_text

def task1b(filepath = 'nytimes_data_final.csv'):
    """Build a vocabulary set from the headlines"""
    df = pd.read_csv(filepath)
    headlines = df['text'].apply(lambda x: normalize(x)).tolist()
    tokenized_headlines = word_tokenize(" ".join(headlines))
    unique_tokens = list(set(tokenized_headlines))
    num_unique_tokens = len(unique_tokens)
    print(num_unique_tokens)
    return unique_tokens

def task1c():
    unique_tokens = task1b()
    d = {word:0 for word in unique_tokens}
    counter = Counter(d)
    df = pd.read_csv('nytimes_data_final.csv')
    headlines = df['text'].apply(lambda x: normalize(x)).tolist()
    tokenized_headlines = word_tokenize(" ".join(headlines))

    for word in tokenized_headlines:
        if word in counter:
            counter[word] += 1
    # print(pd.DataFrame.from_records(counter.most_common(100)))
    top_100_df = pd.DataFrame.from_records(counter.most_common(100)).rename(
        columns = {0: 'word', 1: 'count'}
    )
    sns.set_theme(rc={'figure.figsize':(16,8)})
    plot = sns.barplot(top_100_df, x='word', y='count').get_figure()
    plt.xticks(rotation=90)
    plot.savefig('word_frequency.png')

def task1():
    """Runs task1"""
    task1a()
    task1b()

def main():
    """Run the tasks"""
    # task1()
    df = pd.read_csv("text_only.csv")
    headlines = df['text'].tolist()
    n = len(headlines)
    for i in range(n):
        hl = headlines[i]
        if 800 < i < 1100:
            print(i, hl)

if __name__ == "__main__":
    main()