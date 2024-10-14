"""
Utility functions:
-> pad questions
-> parse tensorboard event logs and save into csvs
-> generate statistics of length of questions
-> plot training and validation statistics
-> plot vqa accuracy
-> plot all accuracies
-> predict answers given an image and questions in that image
"""
import collections
import json
import os
import pickle
import string

def preprocess_text(text):
    """
        Converts a string to lower case, removes punctuations.
    """
    text_token_list = text.strip().split(',')
    text  = ' '.join(text_token_list)

    # Remove punctuations
    table = str.maketrans('', '', string.punctuation)
    words = text.strip().split()
    words = [w.translate(table) for w in words]

    # Set to lowercase & drop empty strings
    words = [word.lower() for word in words if word != '' and word != 's']
    
    text  = ' '.join(words)
    return text