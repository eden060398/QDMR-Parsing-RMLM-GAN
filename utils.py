import torch
import random
import math
import csv
import pandas as pd

from config import *


def mask(seqs_tensor, sep_id, mask_id, mask_prob=1, seq_length=SEQ_LENGTH):
    with torch.no_grad():
        input_tensor = seqs_tensor.clone()
        for i in range(seqs_tensor.size(0)):
            start = 0
            while start < seq_length and seqs_tensor[i, start] != sep_id:
                start += 1
            for j in range(start+1, seq_length):
                if random.random() <= mask_prob:
                    input_tensor[i, j] = mask_id

    return input_tensor


def read_input_file(filename, count=None):
    questions = []
    decomps = []
    with open(filename, newline='', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines, None)
        c = 1
        for line in lines:
            if count is not None and c > count:
                break
            c += 1
            if len(line) == 5:
                _, question, decomp, _, _ = line
                decomp = decomp.replace('return', '')
                questions.append(question)
                decomps.append(decomp)
    return questions, decomps


def write_output_files(questions, decomps, predictions, orig_filename="orig.csv", pred_filename="preds.csv"):
    ids = [str(i) for i in range(len(questions))]
    pd.DataFrame({'question_id': ids, 'question_text': questions, 'decomposition': decomps}).to_csv(orig_filename, index=False)

    pd.DataFrame({'decomposition': predictions}).to_csv(pred_filename, index=False)


def prob_func(x, alpha=3, beta=0.5):
    return beta + (1 - beta) * (math.tanh(alpha*x) / math.tanh(alpha))
