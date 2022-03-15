import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, AdamW
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse
from helper import *
import random
import sys
import utils
import functools
print = functools.partial(print, flush=True)

class MaskedLM:
    def __init__(self, masked_lm_version='bert-base-cased', device='cuda'):
        self.tokenizer = BertTokenizer.from_pretrained(masked_lm_version, do_lower_case=masked_lm_version.endswith("uncased"), cache_dir='./pretrained_models')
        config = BertConfig(len(self.tokenizer))
        self.model = BertForMaskedLM(config).to(device)
        
        self.CLS = '[CLS]'
        self.SEP = '[SEP]'
        self.MASK = '[MASK]'
        self.UNK = '[UNK]'
        self.mask_id = self.tokenizer.convert_tokens_to_ids([self.MASK])[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids([self.SEP])[0]
        self.cls_id = self.tokenizer.convert_tokens_to_ids([self.CLS])[0]

    def generate_masked_sent(self, tokens, probs):
        input_ids = []
        label_ids = []
        for token, p in zip(tokens, probs):
            token_idx = self.tokenizer.convert_tokens_to_ids(token)
            if p < 0.15:
                p = p / 0.15
                if p < 0.8:
                    input_ids.append(self.mask_id)
                elif p < 0.9:
                    random_idx = np.random.choice(len(self.tokenizer))
                    # random_token = self.tokenizer.convert_ids_to_tokens(random_idx)
                    input_ids.append(random_idx)
                else:
                    input_ids.append(token_idx)
                label_ids.append(token_idx)
            else:
                input_ids.append(token_idx)
                label_ids.append(-100)
        return input_ids, label_ids

    def generate_data(self, sent, rng=None):
        sent = self.tokenizer.tokenize(sent)
        if rng is None:
            probs = np.random.random(len(sent))
        else:
            probs = rng.random(len(sent))
        masked_sent_ids, output_ids = self.generate_masked_sent(sent, probs)
        return [self.cls_id] + masked_sent_ids + [self.sep_id], [-100] + output_ids + [-100]

    def generate_data_batch(self, sents):
        masked_sent_ids_batch = []
        output_ids_batch = []
        sents = [self.tokenizer.tokenize(sent) for sent in sents]
        batch_max_len = np.max([len(sent) for sent in sents])
        probs_batch = np.random.random((len(sents), batch_max_len))
        for i, sent in enumerate(sents):
            masked_sent_ids, output_ids = self.generate_masked_sent(sent, probs_batch[i])
            masked_sent_ids_batch.append([self.cls_id] + masked_sent_ids + [self.sep_id])
            output_ids_batch.append([-100] + output_ids + [-100])
        return masked_sent_ids_batch, output_ids_batch

        
    def get_batch_loss(self, data_batch, device='cuda'):
        input_ids_batch, output_ids_batch = data_batch
        batch_max_len = np.max([len(input_ids) for input_ids in input_ids_batch])
        input_ids_padded_batch = [input_ids + [self.sep_id for _ in range(batch_max_len - len(input_ids))] for input_ids in input_ids_batch]
        
        attention_mask = [[1 for _ in range(len(input_ids))] + [0 for _ in range(batch_max_len - len(input_ids))] for input_ids in input_ids_batch]
        attention_mask = torch.tensor(attention_mask).to(device)

        input_ids = torch.tensor(input_ids_padded_batch).to(device)
        label_ids = [output_ids + [-100 for _ in range(batch_max_len - len(output_ids)) ] for output_ids in output_ids_batch]
        label_ids = torch.tensor(label_ids).to(device)

        batch_token_count = np.sum([len([1 for idx in output_ids if idx != -100]) for output_ids in output_ids_batch])

        loss = self.model(input_ids, labels=label_ids, attention_mask=attention_mask)[0]
        return loss, batch_token_count

    def get_loss(self, data, batch_size, device='cuda'):
        masked_sent_ids_all, output_ids_all = data

        total_loss = 0
        total_token_count = 0

        for data_batch in zip(get_batches(masked_sent_ids_all, batch_size), get_batches(output_ids_all, batch_size)):
            loss, batch_token_count = self.get_batch_loss(data_batch, device=device)
            total_loss += loss.item()*batch_token_count
            total_token_count += batch_token_count
        return total_loss/total_token_count


    def sample_output(self, masked_sent_ids, top_k=50, sample=True, device='cuda'):
        input_ids = torch.tensor([masked_sent_ids]).to(device)
        mask_indice = [idx for idx in range(len(masked_sent_ids)) if masked_sent_ids[idx] == self.mask_id]
        logits = self.model(input_ids)[0]
        logits = logits[0, mask_indice, :]

        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist()

def load_lines(path):
    """
    Load the text corpus into a list of lines. Each line is a string.
    """
    lines = open(path).readlines()
    lines = [line.strip() for line in lines]  
    return lines


def generate_dev_data(model, dev_lines, seed=0):
    rng = np.random.RandomState(seed=seed)
    masked_sent_ids_all = []
    output_ids_all = []
    for sent in dev_lines:
        masked_sent_ids, output_ids = model.generate_data(sent, rng=rng)
        masked_sent_ids_all.append(masked_sent_ids)
        output_ids_all.append(output_ids)
    return [masked_sent_ids_all, output_ids_all]


def detokenize(sent):
    """
    Undo wordpiece tokenization. 
    """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            if i > 0:
                new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
            else:
                new_sent.append(tok[2:])
        else:
            new_sent.append(tok)
    return new_sent


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--report', type=int, default=200, help='Frequency of evaluating validation data')
parser.add_argument('--sample_every', type=int, default=None, help='Frequency of sampling completions for test examples after number of training batches')
parser.add_argument('--valid_every', type=int, default=None, help='Frequency of validating and saving model parameters after number of training batches')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--do_train', action='store_true', help='Whether to train the model')
parser.add_argument('--do_test', action='store_true', help='Whether to test the model on generating completions.')
parser.add_argument('--train_data', type=str, help='Path to training data.')
parser.add_argument('--dev_data', type=str, help='Path to development data.')
parser.add_argument('--restore_from', type=str, default=None, help='Path to restoring from a previous model checkpoint.')
parser.add_argument('--save_path', type=str, default='mlm_model.params', help='Path to saving model checkpoint.')
parser.add_argument('--batch_size', type=int, default=1, help="Size of a training batch.")
parser.add_argument('--early_stopping_threshold', type=int, default=2, help='Threshold for early stopping during training.')

args = parser.parse_args()


RANDOM_SEED = args.seed if args.seed is not None else int(np.random.random()*10000)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = torch.nn.CrossEntropyLoss()

model = MaskedLM(device=device)

if args.restore_from is not None:
    checkpoint = torch.load(args.restore_from)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    print("Load from: {}".format(args.restore_from), file=sys.stderr)


if args.do_train:
    # Path to save the newly trained model
    CHECKPOINT_PATH = args.save_path

    # Print out training settings
    print('Training batch size: {}'.format(args.batch_size), file=sys.stderr)
    print('Learning rate: {}'.format(args.lr), file=sys.stderr)
    print('Model path: {}'.format(CHECKPOINT_PATH), file=sys.stderr)

    # Set the learning rate of the optimizer
    optimizer = AdamW(model.model.parameters(), lr=args.lr)

    # Load train and dev data
    train_lines = load_lines(args.train_data)
    dev_lines = load_lines(args.dev_data)

    train_lines = [TreebankWordDetokenizer().detokenize(line.split()) for line in train_lines]
    dev_lines = [TreebankWordDetokenizer().detokenize(line.split()) for line in dev_lines]

    dev_data = generate_dev_data(model, dev_lines, seed=0)


    for masked_sent_ids in dev_data[0][:3]:
        print(TreebankWordDetokenizer().detokenize(detokenize(model.tokenizer.convert_ids_to_tokens(masked_sent_ids[1:-1]))))

    if args.restore_from:
        model.model.eval()
        best_validation_loss = model.get_loss(dev_data, batch_size=args.batch_size)
        model.model.train()
        print('resume training; validation loss: {}'.format(best_validation_loss))
    else:
        best_validation_loss = np.Inf

    starting_epoch = checkpoint['epoch'] + 1 if (args.restore_from is not None) else 0
    n_epochs = args.epochs
    no_improvement_count = checkpoint['no_improvement_count'] if (args.restore_from is not None) else 0
    VALID_EVERY = None if ((args.valid_every is None) or (args.valid_every < 1)) else args.valid_every
    SAMPLE_EVERY = None if ((args.sample_every is None) or (args.sample_every < 1)) else args.sample_every

    early_stopping_counter = utils.EarlyStopping(best_validation_loss=best_validation_loss, no_improvement_count=no_improvement_count, threshold=args.early_stopping_threshold)


    for epoch in range(starting_epoch, n_epochs):
        np.random.shuffle(train_lines)

        count = 0  # cumulative count of training examples
        batch_count = 0 # cumulative count of training batches

        for line_batch in get_batches(train_lines, args.batch_size):
            optimizer.zero_grad()

            input_ids_batch, output_ids_batch = model.generate_data_batch(line_batch)
            loss, batch_token_count = model.get_batch_loss([input_ids_batch, output_ids_batch])
            loss.backward()
            optimizer.step()

            count += len(input_ids_batch)
            batch_count += 1       

            if batch_count > 0 and batch_count % args.report == 0:
                print('Epoch {:.3f} loss: {}'.format(epoch + count/len(train_lines), loss.item()))

            if SAMPLE_EVERY is not None:
                if batch_count > 0 and batch_count % args.sample_every == 0:
                    model.model.eval()
                    with torch.no_grad():
                        masked_sent_ids = input_ids_batch[0]
                        idx_list = model.sample_output(masked_sent_ids)
                        print(TreebankWordDetokenizer().detokenize(detokenize(model.tokenizer.convert_ids_to_tokens(masked_sent_ids[1:-1]))))
                        print(' '.join(model.tokenizer.convert_ids_to_tokens(idx_list)))

                    model.model.train()

            if VALID_EVERY is not None:
                if batch_count > 0  and batch_count % VALID_EVERY == 0:
                    model.model.eval()
                    with torch.no_grad():
                        validation_loss = model.get_loss(dev_data, batch_size=args.batch_size)
                    print('Epoch {:.3f} validation loss: {}'.format(epoch + count/len(train_lines), validation_loss))

                    is_early_stop = early_stopping_counter.check_stopping_criterion(validation_loss)
                    if is_early_stop:
                        print('Validation loss increases for {} epochs in a row.'.format(early_stopping_counter.counter))
                        print('EARLY STOPPING...')
                        sys.exit()

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        print("new best... saving model to {}".format(CHECKPOINT_PATH))
                        torch.save(
                            {'epoch': epoch,
                            'no_improvement_count': early_stopping_counter.counter,
                            'model_state_dict': model.model.state_dict(),
                            'loss': validation_loss},
                            CHECKPOINT_PATH)

                    model.model.train()


        # Validation after each epoch
        model.model.eval()

        with torch.no_grad():
            validation_loss = model.get_loss(dev_data, batch_size=args.batch_size)
        print('Epoch', epoch, 'validation loss:', validation_loss)

        is_early_stop = early_stopping_counter.check_stopping_criterion(validation_loss)
        if is_early_stop:
            print('Validation loss increases for {} epochs in a row.'.format(early_stopping_counter.counter))
            print('EARLY STOPPING...')
            sys.exit()
            break

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print("new best... saving model to {}".format(CHECKPOINT_PATH))
            torch.save(
                {'epoch': epoch,
                'no_improvement_count': early_stopping_counter.counter,
                'model_state_dict': model.model.state_dict(),
                'loss': validation_loss},
                CHECKPOINT_PATH)
            
        model.model.train()
