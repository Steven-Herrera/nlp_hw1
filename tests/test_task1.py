"""Module for testing the hw1.py script"""

import pandas as pd
import re
import sys

sys.path.append("./")
from hw1 import _return_pattern, task1a, task1b, task1c

def test_task1a():
    """Testing regex output"""
    task1a()

def test_task1b():
    task1b()

def test_task1c():
    task1c()

if __name__ == "__main__":
    # test_task1a()
    test_task1c()