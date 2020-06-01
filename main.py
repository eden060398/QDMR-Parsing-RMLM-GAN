from train import train_gan
from eval import eval_model
from models import *
from config import *

if __name__ == '__main__':
    model = train_gan('/Break-dataset/QDMR/train.csv')
    model.save_internal(seq2seq_path='model.dat')

    eval_model([('model.dat', RobertaDecomposer, SEQ_LENGTH)],
               '/Break-dataset/QDMR/dev.csv',
               orig_filenames=["orig.csv"],
               pred_filenames=["pred.csv"])
