import logging
import time
import torch_xla.core.xla_model as xm
from torch.utils.data import TensorDataset, DataLoader

from models import *
from config import *
from utils import *


def train_gan(filename, gen_warmup=5, desc_warmup=1, gan_epochs=5):
    logger = logging.getLogger('myLogger')
    logger.setLevel(logging.INFO)

    logger.info('Starting.')

    device = xm.xla_device()
    print(device, "will be used.")

    print('Initializing model...')
    model = RobertaGAN(SEQ_LENGTH, device)

    print('Reading data file...')
    questions, decomps = read_input_file(filename)

    print('Encoding questions and decompositions...')
    labels = model.seq2seq.encode(questions, decomps)

    print('Optimizing model...')
    sep_id = model.seq2seq.sep_id
    mask_id = model.seq2seq.mask_id

    model = model.to(device)
    model.train()
    seq2seq_opt = torch.optim.Adam(model.seq2seq.parameters(), lr=1e-4)
    start_time = time.time()
    for epoch in range(1, gen_warmup + 1):
        inputs = mask(labels, sep_id, mask_id, mask_prob=prob_func((epoch - 1) / (gen_warmup - 1), beta=0.8))
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        print('Gen Warmup Epoch', epoch, 'starting...')
        seq2seq_loss = model.train_model(loader, seq2seq_opt=seq2seq_opt, train_who=TRAIN_GENERATOR)
        logger.info('S Epoch {e}, Loss={l}'.format(e=epoch, l=seq2seq_loss))
        print('Gen Warmup Epoch', epoch, 'finished. Model saved.', 'Avg Seq2Seq Loss =', seq2seq_loss)

    inputs = mask(labels, sep_id, mask_id)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    classifier_opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-6)
    for epoch in range(1, desc_warmup + 1):
        print('Desc Warmup Epoch', epoch, 'starting...')
        classifier_loss = model.train_model(loader, classifier_opt=classifier_opt, train_who=TRAIN_DESCRIMINATOR)
        logger.info('C Epoch {e}, Loss={l}'.format(e=epoch, l=classifier_loss))
        logger.handlers[0].flush()
        print('Desc Warmup Epoch', epoch, 'finished. Model saved.', 'Avg Classifier Loss =', classifier_loss)

    seq2seq_opt = torch.optim.Adam(model.seq2seq.parameters(), lr=1e-5)
    loader = DataLoader(dataset, batch_size=GAN_BATCH_SIZE, shuffle=True, drop_last=True)
    for epoch in range(1, gan_epochs + 1):
        print('GAN Epoch', epoch, 'starting...')
        seq2seq_loss, classifier_loss = model.train_model(loader, seq2seq_opt, classifier_opt)
        model.save_internal('drive/My Drive/RobertaGAN/high/model_s_{e}.dat'.format(e=epoch),
                            'drive/My Drive/RobertaGAN/high/model_c_{e}.dat'.format(e=epoch))
        logger.info('GAN S Epoch {e}, Loss={l}'.format(e=epoch, l=seq2seq_loss))
        logger.info('GAN C Epoch {e}, Loss={l}'.format(e=epoch, l=classifier_loss))
        logger.handlers[0].flush()
        print('GAN Epoch', epoch, 'finished. Model saved.', 'Avg Seq2Seq Loss =', seq2seq_loss, 'Avg Classifier Loss =',
              classifier_loss)

    print('Finished execution. Time elapsed:', time.time() - start_time, 's')

    return model
