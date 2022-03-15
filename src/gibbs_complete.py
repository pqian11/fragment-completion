import math
import time
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from InfillingModels import RerankingLM
from helper import formatted_response
import argparse
import sys
import functools

from helper import filter_data
print = functools.partial(print, flush=True)


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


def get_blank_config(stimulus, max_nGap=10):
    """
    Randomly sample a number from [1, max_nGap] to specify length of each blank in the stimulus.
    """
    context = ''
    tokens = stimulus.split('%')
    for i, token in enumerate(tokens):
        if i % 2 == 0:
            context += token
        else:
            nGap = np.random.choice(range(1, max_nGap+1), 1, replace=True)[0]
            context += '%{}%'.format(nGap)
    return context


class GibbsComplete:
    def __init__(self, masked_lm_version='bert-base-cased', restore_from=None, device='cuda'):
        # Load pre-trained tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(masked_lm_version, do_lower_case=masked_lm_version.endswith("uncased"), cache_dir='./pretrained_models')

        if restore_from is None:
            # Load pre-trained model
            self.masked_lm = BertForMaskedLM.from_pretrained(masked_lm_version, cache_dir='./pretrained_models').to(device)
            print('Loading masked LM from pretrained BERT ({})'.format(masked_lm_version), file=sys.stderr)
        else:
            # Load trained model
            config = BertConfig(len(self.tokenizer))
            self.masked_lm = BertForMaskedLM(config).to(device)
            checkpoint = torch.load(restore_from)
            self.masked_lm.load_state_dict(checkpoint["model_state_dict"])
            print('Loading masked LM from {}'.format(restore_from), file=sys.stderr)
        self.masked_lm.eval()

        self.CLS = '[CLS]'
        self.SEP = '[SEP]'
        self.MASK = '[MASK]'
        self.mask_id = self.tokenizer.convert_tokens_to_ids([self.MASK])[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids([self.SEP])[0]
        self.cls_id = self.tokenizer.convert_tokens_to_ids([self.CLS])[0]

    def tokens_to_ids_batch(self, batch):
        return [self.tokenizer.convert_tokens_to_ids(sent) for sent in batch]


    def ids_to_tokens_batch(self, batch):
        return [self.tokenizer.convert_ids_to_tokens(sent) for sent in batch]


    def detokenize(self, sent):
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


    def get_init_text_for_context_batch(self, context_batch):
        """ 
        Get initial sentence by padding the blanks with specific number of masks.
        Given a batch of contexts with different blank length configuration.
        """
        initialized_batch = []
        target_indice_batch = []

        for context in context_batch:
            text_template = []
            text_template += [self.CLS]
            fragments = context.strip().split('%')
            for i, fragment in enumerate(fragments):
                if fragments == '':
                    continue
                if i % 2 == 0:
                    text_template += self.tokenizer.tokenize(fragment.strip()) #fragment.strip().split()
                else:
                    nSlot = int(fragment)
                    text_template += [self.MASK] * nSlot
            text_template += [self.SEP]
            target_indice = [i for i, w in enumerate(text_template) if w == self.MASK]

            initialized_batch.append(text_template)
            target_indice_batch.append(target_indice)

            # print(text_template)
        return self.tokens_to_ids_batch(initialized_batch), target_indice_batch   


    def filter_sents(self, sents, contexts):
        _, target_indice_batch = self.get_init_text_for_context_batch(contexts)
        filtered_sents = []
        stop_tokens = set(['[UNK]'])

        for sent_index, sent in enumerate(sents):
            target_indice = target_indice_batch[sent_index]

            # Find all the indice at the intitial position of each blank
            blank_initial_indice = [target_indice[0]]
            for k in range(1, len(target_indice)):
                if target_indice[k] > target_indice[k-1] + 1:
                    blank_initial_indice.append(target_indice[k])
            blank_initial_indice_set = set(blank_initial_indice)

            flag = True
            for i in target_indice:
                token = sent[i]
                if token in stop_tokens:
                    flag = False
                    break
                # The token at the start of a blank should be word initial.
                if i in blank_initial_indice_set and token.startswith('##'):
                    flag = False
                    break
            
            if flag:
                filtered_sents.append(sent)
        return filtered_sents


    @staticmethod
    def extract_answers(sent, target_indice):
        answers = []
        fragment = []
        last_idx = None
        for k, idx in enumerate(target_indice):
            if last_idx is None or (idx - last_idx) == 1:
                fragment.append(sent[idx])
            else:
                answers.append(TreebankWordDetokenizer().detokenize(detokenize(fragment)))
                fragment = [sent[idx]]
            last_idx = idx
        answers.append(TreebankWordDetokenizer().detokenize(detokenize(fragment)))
        return answers
        

    def filter(self, sent, context):
        _, target_indice_batch = self.get_init_text_for_context_batch([context])

        stop_tokens = set(['[UNK]'])

        target_indice = target_indice_batch[0]

        # Find all the indice at the intitial position of each blank
        blank_initial_indice = [target_indice[0]]
        for k in range(1, len(target_indice)):
            if target_indice[k] > target_indice[k-1] + 1:
                blank_initial_indice.append(target_indice[k])
        blank_initial_indice_set = set(blank_initial_indice)

        flag = True
        for i in target_indice:
            token = sent[i]
            if token in stop_tokens:
                flag = False
                break
            # The token at the start of a blank should be word initial.
            if i in blank_initial_indice_set and token.startswith('##'):
                flag = False
                break

        answers = self.extract_answers(sent, target_indice)
            
        return flag, answers


    @staticmethod
    def generate_from_logits(logits_batch, gen_idx_batch, temperature=None, top_k=0, sample=False, return_list=True):
        """ 
        Generate a token at the specific position.
        
        args:
            - logits_batch (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx_batch (list of int): list of locations for which to generate for
            - top_k (int): if > 0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k 
        """
        batch_size = len(logits_batch)
        logits = logits_batch[torch.arange(batch_size), gen_idx_batch]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx


    def iterative_generation(self, contexts, top_k=0, temperature=None, max_iter=300, burnin=200,
                                   device='cuda', print_every=10, verbose=True):
        """ 
        Given a list of contexts of potentially different blank configuration.
        Generate for one random position at a timestep.
        """
        ids_batch, target_indice_batch = self.get_init_text_for_context_batch(contexts)
        batch_size = len(ids_batch)
        batch_max_len = np.max([len(ids) for ids in ids_batch])
        
        for ii in range(max_iter):
            selected_index_batch = [np.random.choice(target_indice) for target_indice in target_indice_batch]
            for jj in range(batch_size):
                kk = selected_index_batch[jj]
                ids_batch[jj][kk] = self.mask_id

            # Add padding to the input
            ids_padded_batch = torch.tensor([ids + [self.sep_id]*(batch_max_len - len(ids)) for ids in ids_batch]).to(device)

            # Attention mask
            attention_mask = [[1 for _ in range(len(ids))] + [0 for _ in range(batch_max_len - len(ids))] for ids in ids_batch]
            attention_mask = torch.tensor(attention_mask).to(device)

            output = self.masked_lm(ids_padded_batch, attention_mask=attention_mask)
            logits_batch = output[0]
            topk = top_k if (ii >= burnin) else 0
            idxs = self.generate_from_logits(logits_batch, gen_idx_batch=selected_index_batch, top_k=topk, temperature=temperature, sample=(ii < burnin))

            for jj, kk in enumerate(selected_index_batch):
                ids_batch[jj][kk] = idxs[jj]
                
            if verbose and np.mod(ii+1, print_every) == 0:
                for_print = self.tokenizer.convert_ids_to_tokens(ids_batch[0])
                for_print = for_print[:kk+1] + ['(*)'] + for_print[kk+1:]
                print("iter", ii+1, " ".join(for_print))
        return self.ids_to_tokens_batch(ids_batch)


    def infill(self, stimulus, nSample=200, batch_size=50, nOutput=35, top_k=100, temperature=1.0, burnin=250, max_iter=500, do_rerank=True, rerank_lm=None, rerank_metric='avg_word_surprisal', device='cuda', verbose=False, print_every=1):
        """
        Infill incomplete sentence.
        """
        if do_rerank:
            assert rerank_lm is not None
            assert nOutput is not None

        # Whether the number of tokens in each blank is specified
        blank_config_spec = True
        tokens = stimulus.split('%')
        if tokens[1] == '':
            blank_config_spec = False

        sentences = []
        answers_all = []

        n_batches = math.ceil(nSample / batch_size)
        start_time = time.time()
        for batch_n in range(n_batches):
            # Specify contexts
            if blank_config_spec:
                contexts = [stimulus for _ in range(batch_size)]        
            else:
                contexts = [get_blank_config(stimulus) for _ in range(batch_size)]
                if verbose:
                    for context in contexts:
                        print(context)

            # Run interative generation
            batch = self.iterative_generation(contexts, top_k=top_k, temperature=temperature, 
                                                    burnin=burnin, max_iter=max_iter, device=device, verbose=False)
            if (batch_n + 1) % print_every == 0:
                print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time), file=sys.stderr)
                start_time = time.time()

            for sent, context in zip(batch, contexts):
                flag, answers = self.filter(sent, context)
                if flag:
                    sentences.append(sent)
                    answers_all.append(answers)

        completions = [TreebankWordDetokenizer().detokenize(detokenize(sent[1:-1])) for sent in sentences]

        if do_rerank:
            reranked_completions_with_scores = rerank_lm.rerank_samples(completions, metric=rerank_metric, detokenized=True)
            return [completion for _, completion, _ in reranked_completions_with_scores[:nOutput]], [answers_all[completion_idx] for completion_idx, _, _ in reranked_completions_with_scores[:nOutput]]
        else:
            return completions, answers_all


parser = argparse.ArgumentParser(description='Fill in the blanks in incomplete sentences.')
parser.add_argument('--fpath', type=str, default=None, help='File path to a list of incomplete sentences.')
parser.add_argument('--rerank', action='store_true', help='Whether to rerank completions')
parser.add_argument('--filter_repeat', action='store_true', help='Whether to filter repeated completions')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling completions.')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--top_k', type=int, default=50, help='Sample from top k.')
parser.add_argument('--n_sample', type=int, default=500, help='Number of samples')
parser.add_argument('--n_output', type=int, default=35, help='Number of output if reranking the candidate completions.')
parser.add_argument('--burnin', type=int, default=250, help='Number of iteration during the burnin period.')
parser.add_argument('--max_iter', type=int, default=500, help='Maximum number of iteration.')
parser.add_argument('--batch_size', type=int, default=50, help="Size of a training batch.")
parser.add_argument('--output_path', type=str, default=None, help="Path to save the output answers.")
parser.add_argument('--restore_from', type=str, default=None, help='Path to trained masked LM model.')
parser.add_argument('--restore_rerank_lm_from', type=str, default=None, help='Path to trained reranking LM model.')

args, unknown = parser.parse_known_args()

RANDOM_SEED = args.seed if args.seed is not None else int(np.random.random()*10000)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

N_SAMPLE = args.n_sample
BATCH_SIZE= args.batch_size
TOP_K = args.top_k
TEMPERATURE = args.temperature

BURNIN = args.burnin
DO_SAMPLE = True
MAX_ITER = args.max_iter

DO_RERANK = args.rerank

N_OUTPUT = args.n_output

OUTPUT_PATH = args.output_path if (args.output_path is not None) else None

DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model
model = GibbsComplete(restore_from=args.restore_from, device=DEVICE)

if DO_RERANK:
    # Load reranking language model
    rerank_lm = RerankingLM(restore_from=args.restore_rerank_lm_from, device=DEVICE)

print("GibbsComplete:\nSampling temperature: {}\nNumer of samples: {}\nBatch size: {}".format(TEMPERATURE, N_SAMPLE, BATCH_SIZE), file=sys.stderr)
print("Burn-in: {}\nNumber of iteration: {}\nTop-k: {}".format(BURNIN, MAX_ITER, TOP_K), file=sys.stderr)
print("Rerank: {}\nNumber of reranking output: {}".format(DO_RERANK, N_OUTPUT), file=sys.stderr)

if args.fpath is not None:
    with open(args.fpath) as f:
        stimuli = f.readlines()
    stimuli = [stimulus.strip() for stimulus in stimuli if stimulus.strip() != '']
else:
    stimuli = ["%9% insisted would %9%.",
                "%% violin %% applauded %%.",
                "%% invited %% beer %%.",
                "%% baked %% flowers %%.",
                "%% drank %% newspapers %%."
                ]

if OUTPUT_PATH is not None:
    outf = open(OUTPUT_PATH, 'w')

with torch.no_grad():
    for stimulus in stimuli:
        print(stimulus)
        completions, answers_all = model.infill(stimulus, rerank_lm=rerank_lm, do_rerank=DO_RERANK, rerank_metric='avg_word_surprisal', 
                    nSample=N_SAMPLE, batch_size=BATCH_SIZE, nOutput=N_OUTPUT, top_k=TOP_K, temperature=TEMPERATURE, 
                    max_iter=MAX_ITER, burnin=BURNIN, device=DEVICE)
        
        for completion in completions:
            print(completion)
        print()

        if OUTPUT_PATH is not None:
            outf.write(stimulus+'\n')
            outf.writelines(['\t'.join(answers)+'\n' for answers in answers_all])
            outf.write('\n')

if OUTPUT_PATH is not None:
    outf.close()

print("Successfully finished!", file=sys.stderr)   
    

