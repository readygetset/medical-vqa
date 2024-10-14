from utils import preprocess_text

import argparse
import collections
import json
import os
import pickle
import string
import random

def preprocess_SLAKE():
    train_file = "train.json"
    val_file = "validate.json"
    test_file = "train.json"
    output_file = "SLAKE.txt"
    
    train_q = json.load(open(train_file, 'r', encoding='utf-8'))
    val_q = json.load(open(val_file, 'r', encoding='utf-8'))
    test_q = json.load(open(test_file, 'r', encoding='utf-8'))
    questions = train_q + val_q + test_q

    count = 0
    with open(output_file, 'w') as out:
        for question in questions:
            if question['q_lang']=='en' and question['base_type']=='vqa':
                count += 1
                image_id = question['img_name']
                ques = preprocess_text(question['question']).replace(',','')
                answer = preprocess_text(question['answer']).replace(',','')
                if answer != '':
                    out.write(str(image_id) + "," + ques + "," + answer + "\n")
                
def preprocess_RAD(data_dir):
    output_file = "VQA_RAD.txt"
    questions = json.load(open(data_dir, 'r'))

    count = 0
    with open(output_file, 'w') as out:
        for question in questions:
            count += 1
            image_id = question['image_name']
            ques = preprocess_text(question['question']).replace(',','')
            answer = preprocess_text(str(question['answer'])).replace(',','')
            if answer != '':
                out.write(str(image_id) + "," + ques + "," + answer + "\n")

def preprocess_2019(data_dir):
    with open(data_dir, 'r', encoding='UTF-8') as infile, open('VQA_2019.txt', 'w', encoding='UTF-8') as outfile:
        for line in infile:
            line = line.replace(',', '')
            line = line.replace('|', ',')
            outfile.write(line)

def combine_files(file1, file2, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(file1, 'r', encoding='utf-8') as infile1:
            outfile.write(infile1.read())
        outfile.write('\n')
        with open(file2, 'r', encoding='utf-8') as infile2:
            outfile.write(infile2.read())

def split_data(input_file, train_file, val_file, val_ratio=0.1):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    random.shuffle(lines)
    val_size = int(len(lines) * val_ratio)
    val_data = lines[:val_size]
    train_data = lines[val_size:]
    with open(train_file, 'w', encoding='utf-8') as train_out:
        train_out.writelines(train_data)
    with open(val_file, 'w', encoding='utf-8') as val_out:
        val_out.writelines(val_data)


def save_answer_freqs():
    """
        Reads the preprocessed train_data.txt file in data_dir, calculates the
        frequences of answers, and saves them in answers_freqs.pkl file
    """
    with open('train_data.txt', 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')

    answers = [x.split(',')[2].strip() for x in data if x!='']
    answer_freq = dict(collections.Counter(answers))
    print(f"Total number of answers in training data - {len(answer_freq)}!")
    
    with open('answers_freqs.pkl', 'wb') as f:
        pickle.dump(answer_freq, f)

    print("Saving answer frequencies from train data!")

def save_vocab_questions(min_word_count = 3):
    """
        Reads the preprocessed train_data.txt file in data_dir and saves the vocabulary
        of words for the questions in questions_vocab.pkl file
    """
    with open('train_data.txt', 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')

    questions           = [x.split(',')[1].strip() for x in data]
    words               = [x.split() for x in questions]
    words               = [w for x in words for w in x]
    word_index          = [w for (w, freq) in collections.Counter(words).items() if freq > min_word_count]
    word_index          = {w: i+2 for i, w in enumerate(word_index)}
    word_index["<pad>"] = 0
    word_index["<unk>"] = 1
    index_word          = {i: x for x, i in word_index.items()}
    print(f"Total number of words in questions in training data - {len(word_index)}!")

    with open('questions_vocab.pkl', 'wb') as f:
        pickle.dump({"word2idx": word_index, "idx2word": index_word}, f)

    print("Saving vocab for words in questions!")

def main():
    preprocess_SLAKE()
    #preprocess_RAD('RawVQARAD.json')
    #preprocess_2019('RawVQA2019.txt')
    #combine_files('VQA_2019.txt', 'VQA_RAD.txt', 'dataset.txt')
    combine_files('dataset_1.txt', 'SLAKE.txt', 'dataset.txt')
    split_data('dataset.txt', 'train_data.txt', 'val_data.txt', val_ratio=0.1)
    save_answer_freqs()
    save_vocab_questions()

if __name__ == '__main__':
    main()