"""A module for completing Task 1 of Homework 1 for CSCI B651 Natural Language Processing

Task 1: Regex and building vocabulary of a corpus

TODO:
    [] Build a regex pattern to detect all the headlines that mention all the pronoun names (Donald Trump,
       Joe Biden, Thai Le, Indiana University Bloomington) in the dataset. List all the headlines that you can
       find and print out how many unique names you can collect.

    [] Build a set of vocabulary from all the headlines. This can be done through word tokenization, text
       normalization--e.g., case folding, lemmatization. Remember, a set of vocabulary contains only unique
       types of words. Please print out the size of your vocabulary.

    [] Count the frequency of each vocabulary across all the headlines. Plot a plot showing such frequency
       distribution for the top 100 most frequent words: the x-axis represents each word, and the y-axis
       represents the frequency of the corresponding word.

CITATIONS:
    @misc{nlp-hw1,
    author = {Andy Boothe},
    title = {popular-names-by-country-dataset},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {url{https://github.com/sigpwned/popular-names-by-country-dataset}},
    commit = {eb62e13d4d62dd96cdfae79d293a02066352205f}
    }"""

import pandas as pd
import re

"""Task 1a
I'm thinking of building a regex to combine different first/last names together.
Then I'll have to make a disgusting regex like fname1|lname1|fname1|lname2 ...
I'm gonna have to make a function to return a match or None
"""
def _get_combinations(col="Romanized Name"):
    """Combines each first name with every other last name
    to return a list of full names"""
    fnames_df = pd.read_csv("name_data/first_names.csv")
    lnames_df = pd.read_csv('name_data/last_names.csv')

    full_names = []
    for fname in fnames_df[col]:
        fname_regex = _make_regex_name(fname)
        for lname in lnames_df[col]:
            lname_regex = _make_regex_name(lname)
            full_name = f"{fname_regex} {lname_regex}"
            full_names.append(full_name)
    return full_names

def _make_regex_name(name):
    """Converts a name into a regex to match upper and lowercase names"""
    first_letter = name[0]
    truncated_name = name[1:]
    opposite_case = first_letter.lower() if first_letter.isupper() else first_letter.upper()
    regex_name = f"[{first_letter}{opposite_case}]{truncated_name}"
    return regex_name

def _make_regex(full_names):
    """Make a regex using all of the names returned from `_get_combinations`"""
    pattern = "|".join(full_names)
    return pattern

def task1():
    """RUns task1"""
    full_names = _get_combinations()
    pattern = _make_regex(full_names)
    return pattern

def main():
    """Run the tasks"""
    task1()

if __name__ == "__main__":
    main()