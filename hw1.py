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
    task1c()

def levenshtein_full_matrix(str1, str2):
    """Calculates the Levenshtein distance between two strings
    
    Citation:
        GeeksForGeeks
        31 Jan 2024
        Introduction to Levenshtein distance
        Source Code
        https://www.geeksforgeeks.org/introduction-to-levenshtein-distance/
    """
    m = len(str1)
    n = len(str2)
 
    # Initialize a matrix to store the edit distances
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
 
    # Initialize the first row and column with values from 0 to m and 0 to n respectively
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
 
    # Fill the matrix using dynamic programming to compute edit distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                # Characters match, no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Characters don't match, choose minimum cost among insertion, deletion, or substitution
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
 
    # Return the edit distance between the strings
    return dp[m][n]

def suggest(char, k):
    with open('WordList-1.txt', 'r', encoding='latin-1') as txt:
        word_lst = txt.read().splitlines()
    
    last_substr = char.split()[-1]
    sub_lst = [word for word in word_lst if word.startswith(last_substr)]

    distances = {}
    for word in sub_lst:
        levenshtein_distance = levenshtein_full_matrix(last_substr, word)
        distances[word] = levenshtein_distance
    
    sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))

    top_words = {}
    for i, tup in enumerate(sorted_distances.items()):
        top_words[tup[0]] = tup[1]
        if i >= k - 1:
            break

    return top_words

def main():
    """Run the tasks"""
    # task1()
    char = "why i"
    k = 5
    top_words = suggest(char, k)
    print(top_words)

if __name__ == "__main__":
    main()