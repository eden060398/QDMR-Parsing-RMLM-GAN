import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead

TRAIN_BOTH = 0
TRAIN_GENERATOR = 1
TRAIN_DESCRIMINATOR = 2

REFERENCES = ['#{i}'.format(i=i) for i in range(1, 31)]


class RobertaDecomposer(nn.Module):
    def __init__(self, seq_length, model_name='roberta-base'):
        super(RobertaDecomposer, self).__init__()

        self.seq_length = seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)

        self.tokenizer.add_tokens(REFERENCES)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.sep_id = self.tokenizer.convert_tokens_to_ids('</s>')
        self.mask_id = self.tokenizer.convert_tokens_to_ids('<mask>')

    def forward(self, inputs, labels=None):
        if labels is None:
            return self.model(inputs)[0]
        else:
            return self.model(inputs, masked_lm_labels=labels)[:2]

    def encode(self, questions, decomps=None):
        if decomps is None:
            seqs = ['{q} </s>'.format(q=q) for q in questions]
        else:
            seqs = ['{q} </s> {d}'.format(q=q, d=d) for q, d in zip(questions, decomps)]

        seqs_enc = [
            self.tokenizer.encode(seq, add_special_tokens=True, max_length=self.seq_length, pad_to_max_length=True)
            for seq in seqs]
        seqs_tensor = torch.tensor(seqs_enc, dtype=torch.long)

        return seqs_tensor

    def decode(self, inputs, outputs):
        seqs = []
        for i, inp in enumerate(inputs):
            j = 0
            while j < inp.size(0) and inp[j] != self.sep_id:
                j += 1
            ids = [idx.item() for idx in outputs[i, j + 1:].argmax(dim=1)]
            seqs.append(self.tokenizer.decode(ids))
        return seqs

    @staticmethod
    def clean(decomps):
        cleaned = []
        for decomp in decomps:
            clean = re.sub(r'\s*((<pad>)|(<s>)|(</s>))\s*', '', decomp)
            clean = re.sub(r'\s+', r' ', clean.strip())

            clean = re.sub(r'([^a-zA-Z]+)\1+', r'\1', clean)
            clean = re.sub(r'(\b\w+\b)(?:\s*\1)+', r'\1', clean).rstrip(' ,;')

            cleaned.append(clean)

        return cleaned


class RobartaClassifier(nn.Module):
    def __init__(self, seq_length, model_name='roberta-base'):
        super(RobartaClassifier, self).__init__()

        self.seq_length = seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.tokenizer.add_tokens(REFERENCES)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.model.config.hidden_size

        self.embeddings = self.model.embeddings.word_embeddings

        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size * self.seq_length, 1),
            nn.Tanh())

    def forward(self, input_embeds):
        last_hidden_state = self.model(inputs_embeds=input_embeds)[0]
        return self.cls(last_hidden_state.view(-1, self.hidden_size * self.seq_length))

    def encode(self, seqs=None, logits=None):
        if seqs is not None:
            seqs_enc = [
                self.tokenizer.encode(seq, add_special_tokens=True, max_length=self.seq_length, pad_to_max_length=True)
                for seq in seqs]
            seqs_tensor = torch.tensor(seqs_enc, dtype=torch.long)
            return self.embeddings(seqs_tensor)
        if logits is not None:
            return torch.tensordot(logits, self.embeddings.weight, dims=1)


class RobertaGAN(nn.Module):
    def __init__(self, seq_length, device):
        super(RobertaGAN, self).__init__()

        self.device = device

        self.seq_length = seq_length

        self.seq2seq = RobertaDecomposer(seq_length, model_name='roberta-base')
        self.classifier = RobartaClassifier(seq_length, model_name='distilroberta-base')

    def _run_seq2seq_batch(self, inputs, labels, calc_d_loss=True):
        ce_loss, raw_logits = self.seq2seq(inputs, labels)
        logits = F.softmax(raw_logits, dim=2)

        if calc_d_loss:
            outputs = self.classifier.encode(logits=logits)

            pred = self.classifier(outputs)
            d_loss = torch.mean(pred)
        else:
            d_loss = torch.zeros(1, dtype=torch.float, device=self.device)

        return d_loss, ce_loss

    def _run_classifier_batch(self, inputs, labels):
        batch_size = inputs.size(0)
        half_batch_size = batch_size // 2

        label_embeds = self.classifier.embeddings(labels[half_batch_size:])

        with torch.no_grad():
            raw_logits = self.seq2seq(inputs).detach()

        outputs = self.classifier.embeddings(raw_logits[:half_batch_size].argmax(dim=2))
        class_inp = torch.cat([outputs, label_embeds])

        rand_idx = torch.randperm(batch_size, device=self.device)

        pred = self.classifier(class_inp[rand_idx])
        d_loss = (torch.mean(pred[rand_idx >= half_batch_size]) - torch.mean(pred[rand_idx < half_batch_size])) / 2

        return d_loss

    def train_model(self, loader, seq2seq_opt=None, classifier_opt=None, train_who=TRAIN_BOTH):
        seq2seq_batch = 0
        classifier_batch = 0
        seq2seq_loss = 0.0
        classifier_loss = 0.0
        only_generator = train_who == TRAIN_GENERATOR
        train_classifier = train_who == TRAIN_DESCRIMINATOR

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            if train_classifier:
                classifier_opt.zero_grad()
                loss = self._run_classifier_batch(x, y)
                loss_val = loss.item()
                loss.backward()
                xm.optimizer_step(classifier_opt, barrier=True)

                del loss

                classifier_loss += loss_val
                classifier_batch += 1
                print('Classifier | Batch', classifier_batch, '| Loss =', loss_val)

            else:
                seq2seq_opt.zero_grad()
                d_loss, ce_loss = self._run_seq2seq_batch(x, y, not only_generator)
                d_val = d_loss.item()
                ce_val = ce_loss.item()

                if only_generator:
                    loss = ce_loss
                else:
                    loss = ce_loss + 0.1 * d_loss
                loss_val = loss.item()
                loss.backward()
                xm.optimizer_step(seq2seq_opt, barrier=True)

                del loss
                del d_loss
                del ce_loss

                seq2seq_loss += ce_val
                seq2seq_batch += 1
                print('Seq2Seq | Batch', seq2seq_batch, '| D Loss =', d_val,
                      '| CE Loss =', ce_val, '| CE Avg =', seq2seq_loss / seq2seq_batch)

            if train_who == TRAIN_BOTH:
                train_classifier = not train_classifier

        if train_who == TRAIN_BOTH:
            return seq2seq_loss / seq2seq_batch, classifier_loss / classifier_batch
        if train_who == TRAIN_GENERATOR:
            return seq2seq_loss / seq2seq_batch
        if train_who == TRAIN_DESCRIMINATOR:
            return classifier_loss / classifier_batch

    def save_internal(self, seq2seq_path=None, classifier_path=None):
        if seq2seq_path is not None:
            xm.save(self.seq2seq.state_dict(), seq2seq_path)
        if classifier_path is not None:
            xm.save(self.classifier.state_dict(), classifier_path)

    def load_internal(self, seq2seq_path=None, classifier_path=None):
        if seq2seq_path is not None:
            self.seq2seq.load_state_dict(torch.load(seq2seq_path))
        if classifier_path is not None:
            self.classifier.load_state_dict(torch.load(classifier_path))
