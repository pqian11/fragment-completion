from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, BartTokenizer, BartConfig, BartForConditionalGeneration, T5Tokenizer, T5Config, T5ForConditionalGeneration, AutoConfig, AdamW
import torch
import argparse
import random
import numpy as np
import sys
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from helper import *
import functools
print = functools.partial(print, flush=True)


class InfillingLM:
    def __init__(self, is_random_init=False, device='cuda'):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='pretrained_models')

        if is_random_init:
            print('Initialize with random weights', file=sys.stderr)
            config = GPT2Config(len(self.tokenizer))
            self.model = GPT2LMHeadModel(config).to(device)
        else:
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='pretrained_models').to(device)

        self.BLANK = '[BLANK]'
        self.FILLER = '[FILLER]'
        self.SEP = '[SEP]'
        self.num_added_tokens = self.tokenizer.add_tokens([self.BLANK, self.FILLER, self.SEP])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.BLANK_id = self.tokenizer.convert_tokens_to_ids(self.BLANK)
        self.FILLER_id = self.tokenizer.convert_tokens_to_ids(self.FILLER)
        self.SEP_id = self.tokenizer.convert_tokens_to_ids(self.SEP)

    def get_batch_loss(self, data_batch, device='cuda'):
        context_tokens_batch = [self.tokenizer.tokenize(' '.join(context)) for context, _ in data_batch]
        answer_tokens_batch = [self.tokenizer.tokenize(' '.join(answer)) for _, answer in data_batch]

        context_ids_batch = [self.tokenizer.convert_tokens_to_ids(context_tokens) for context_tokens in context_tokens_batch]
        answer_ids_batch = [self.tokenizer.convert_tokens_to_ids(answer_tokens) for answer_tokens in answer_tokens_batch]

        infilling_ids_batch = [context_ids + [self.SEP_id] + answer_ids for context_ids, answer_ids in zip(context_ids_batch, answer_ids_batch)]
        batch_max_len = np.max([len(infilling_ids) for infilling_ids in infilling_ids_batch])
        infilling_ids_padded_batch = [infilling_ids + [self.tokenizer.bos_token_id for _ in range(batch_max_len - len(infilling_ids))] for infilling_ids in infilling_ids_batch]
        
        attention_mask = [[1 for _ in range(len(infilling_ids))] + [0 for _ in range(batch_max_len - len(infilling_ids))] for infilling_ids in infilling_ids_batch]
        attention_mask = torch.tensor(attention_mask).to(device)

        input_ids = torch.tensor(infilling_ids_padded_batch).to(device)
        label_ids = [[-100 for _ in range(len(context_ids)+1)] + answer_ids + [-100 for _ in range(batch_max_len - 1 - len(context_ids) - len(answer_ids)) ] for context_ids, answer_ids in zip(context_ids_batch, answer_ids_batch)]
        label_ids = torch.tensor(label_ids).to(device)

        loss = self.model(input_ids, labels=label_ids, attention_mask=attention_mask)[0]
        batch_token_count = np.sum([len(answer_tokens) for answer_tokens in answer_tokens_batch])
        return loss, batch_token_count

    def get_loss(self, data, batch_size, device='cuda'):
        total_loss = 0
        total_token_count = 0

        for data_batch in get_batches(data, batch_size):
            loss, batch_token_count = self.get_batch_loss(data_batch, device=device)
            total_loss += loss.item()*batch_token_count
            total_token_count += batch_token_count

        return total_loss/total_token_count

    def count_blanks(self, context_ids):
        blank_count = 0
        for idx in context_ids:
            if idx == self.BLANK_id:
                blank_count += 1
        return blank_count

    def get_completion(self, sent, add_blank_symbol=True, device='cuda'):
        if add_blank_symbol:
            stimulus = sent
            context = stimulus.replace('%%', self.BLANK)
        else:
            context = sent
            # In case there is percentile symbol in the context
            stimulus = sent.replace('%', ' per centile').replace(self.BLANK, '%%')

        context_tokens = self.tokenizer.tokenize(context + ' ' + self.SEP)
        context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)

        # Count the number of blanks in the incomplete input
        blank_count = self.count_blanks(context_ids)

        answers = []
        prefix_ids = context_ids[:]
        filler_count = 0
        answer_start = len(prefix_ids)

        while filler_count < blank_count and len(prefix_ids) < (len(context_ids) + 100):
            input_ids = torch.tensor(prefix_ids).unsqueeze(0).to(device)
            prediction_scores = self.model(input_ids)[0]
            idx = sample_from_scores(prediction_scores[:, -1])[0]
            token = self.tokenizer.convert_ids_to_tokens(idx)
            prefix_ids.append(idx)
            if token == self.FILLER:
                filler_count += 1
                answer = self.tokenizer.decode(prefix_ids[answer_start:-1])
                answers.append(answer.strip())
                answer_start = len(prefix_ids)

        # Add placeholder FILLERs if there are less number of the answer chunks than that of the blanks 
        if len(answers) < blank_count:
            answers += ['[FILLER]' for _ in range(blank_count - len(answers))]

        print(formatted_response(stimulus, ['____' for _ in range(blank_count)]) + ' [SEP] ' + ''.join([answer + ' [FILLER] ' for answer in answers]))
        return

    def complete(self, stimulus, sampling=True, max_len=50, n_sample=100, top_k=50, top_p=0.92, temperature=1, num_beams=10, device='cuda'):
        context = stimulus.replace('%%', self.BLANK)
        context_tokens = self.tokenizer.tokenize(context + ' ' + self.SEP)
        context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)

        nBlank = self.count_blanks(context_ids)

        input_ids = torch.tensor(context_ids).unsqueeze(0).to(device)
        output_ids_batch = self.model.generate(input_ids, do_sample=sampling, max_length=max_len, 
                                                top_k=top_k, top_p=top_p, temperature=temperature, num_return_sequences=n_sample, pad_token_id=50256)
        samples = [self.tokenizer.convert_ids_to_tokens(output_ids[len(context_ids):]) for output_ids in output_ids_batch]

        answers_batch = []
        for tokens in samples:
            nFiller = 0
            answers = []
            answer_start_index = 0
            for k, token in enumerate(tokens):
                if token == self.FILLER:
                    nFiller += 1
                    answers.append(self.tokenizer.convert_tokens_to_string(tokens[answer_start_index:k]))
                    answer_start_index = k + 1
                if nFiller == nBlank:
                    # completion = formatted_response(stimulus, answers)
                    answers_batch.append(answers)
                    break 
        return answers_batch

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        return

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        return


class InfillingBART:
    def __init__(self, is_random_init=False, device='cuda'):
        self.BLANK = '<mask>'
        self.FILLER = '[FILLER]'
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', cache_dir='pretrained_models')

        if is_random_init:
            print('Initialize with random weights', file=sys.stderr)
            config = BartConfig()
            self.model = BartForConditionalGeneration(config).to(device)
        else:
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', cache_dir='pretrained_models').to(device)

        self.num_added_toks = self.tokenizer.add_tokens([self.FILLER])  # <mask> is in the original vocabulary of BART model
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.BLANK_id = self.tokenizer.convert_tokens_to_ids(self.BLANK)
        self.FILLER_id = self.tokenizer.convert_tokens_to_ids(self.FILLER)

    def get_batch_loss(self, data_batch, device='cuda'):
        context_tokens_batch = [self.tokenizer.tokenize(' '.join(context)) + [self.tokenizer.eos_token] for context, _ in data_batch]
        answer_tokens_batch = [[self.tokenizer.pad_token] + self.tokenizer.tokenize(' '.join(answer)) + [self.tokenizer.eos_token] for _, answer in data_batch]
        
        context_ids_batch = [self.tokenizer.convert_tokens_to_ids(context_tokens) for context_tokens in context_tokens_batch]
        answer_ids_batch = [self.tokenizer.convert_tokens_to_ids(answer_tokens) for answer_tokens in answer_tokens_batch]

        batch_max_len = np.max([len(context_ids) for context_ids in context_ids_batch])
        context_ids_padded_batch = [context_ids + [self.tokenizer.eos_token_id for _ in range(batch_max_len - len(context_ids))] for context_ids in context_ids_batch]
        attention_mask = [[1 for _ in range(len(context_ids))] + [0 for _ in range(batch_max_len - len(context_ids))] for context_ids in context_ids_batch]
        attention_mask = torch.tensor(attention_mask).to(device)

        # List of decoder input/label lengths
        decoder_token_count_batch = [len(answer_tokens) - 1 for answer_tokens in answer_tokens_batch]

        decoder_batch_max_len = np.max(decoder_token_count_batch)
        decoder_input_ids_padded_batch = [answer_ids[:-1] + [self.tokenizer.eos_token_id for _ in range(decoder_batch_max_len - len(answer_ids) + 1)] for answer_ids in answer_ids_batch]
        decoder_label_ids_padded_batch = [answer_ids[1:] + [-100 for _ in range(decoder_batch_max_len - len(answer_ids) + 1)] for answer_ids in answer_ids_batch]
        decoder_attention_mask = [[1 for _ in range(len(answer_ids) - 1)] + [0 for _ in range(decoder_batch_max_len - len(answer_ids) + 1)] for answer_ids in answer_ids_batch]
        decoder_attention_mask = torch.tensor(decoder_attention_mask).to(device)

        input_ids = torch.tensor(context_ids_padded_batch).to(device)
        decoder_input_ids = torch.tensor(decoder_input_ids_padded_batch).to(device)
        decoder_label_ids = torch.tensor(decoder_label_ids_padded_batch).to(device)

        output = self.model(input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_label_ids, 
                        attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, return_dict=True)

        loss = output.loss

        batch_token_count = np.sum(decoder_token_count_batch)
        return loss, batch_token_count

    def get_loss(self, data, batch_size, device='cuda'):
        total_loss = 0
        total_token_count = 0

        for data_batch in get_batches(data, batch_size):
            loss, batch_token_count = self.get_batch_loss(data_batch, device=device)
            total_loss += loss.item()*batch_token_count
            total_token_count += batch_token_count

        return total_loss/total_token_count

    def count_blanks(self, context_ids):
        nBlank = 0
        for idx in context_ids:
            if idx == self.BLANK_id:
                nBlank += 1
        return nBlank

    def get_completion(self, sent, nSample=1, max_len=50, top_k=50, top_p=0.92, add_blank_symbol=True, device='cuda'):
        if add_blank_symbol:
            stimulus = sent
            context = stimulus.replace('%%', self.BLANK)
        else:
            context = sent
            stimulus = sent.replace(self.BLANK, '%%')

        context_tokens = self.tokenizer.tokenize(context)+ [self.tokenizer.eos_token] 
        context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        input_ids = torch.tensor(context_ids).unsqueeze(0).to(device)

        nBlank = self.count_blanks(context_ids)
        
        output_ids = self.model.generate(input_ids, decoder_start_token_id=self.tokenizer.pad_token_id, do_sample=True, max_length=max_len, 
            top_k=top_k, top_p=top_p, temperature=1, num_return_sequences=nSample)
        samples = [self.tokenizer.convert_ids_to_tokens(sample) for sample in output_ids]        

        for tokens in samples:
            tokens = tokens[1:] # the first token is the pad token
            nFiller = 0
            answers = []
            answer_start_index = 0
            for k, token in enumerate(tokens):
                if token == self.FILLER:
                    nFiller += 1
                    answers.append(self.tokenizer.convert_tokens_to_string(tokens[answer_start_index:k]))
                    answer_start_index = k + 1
                if nFiller == nBlank:
                    break
            print(formatted_response(stimulus, ['____' for _ in range(nBlank)]) + '  ' + ''.join([answer + ' {} '.format(self.FILLER) for answer in answers]))
        return

    def complete(self, stimulus, sampling=True, max_len=50, n_sample=100, top_k=50, top_p=0.92, temperature=1, num_beams=10, device='cuda'):
        """
        Assume the input stimulus uses %% as placeholders for blanks.
        Return generated completions to the incomplete stimulus.
        """
        context = stimulus.replace('%%', self.BLANK)
        context_tokens = self.tokenizer.tokenize(context) + [self.tokenizer.eos_token]
        context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)

        nBlank = self.count_blanks(context_ids)

        input_ids = torch.tensor(context_ids).unsqueeze(0).to(device)

        if not sampling:
            output_ids_batch = self.model.generate(input_ids, decoder_start_token_id=self.tokenizer.pad_token_id, num_beams=10, min_length=1, max_length=max_len, 
                                                    num_return_sequences=n_sample, early_stopping=True)
            samples = [self.tokenizer.convert_ids_to_tokens(output_ids) for output_ids in output_ids_batch]
        else:
            output_ids_batch = self.model.generate(input_ids, decoder_start_token_id=self.tokenizer.pad_token_id, do_sample=sampling, max_length=max_len, 
                                                    top_k=top_k, top_p=top_p, temperature=temperature, num_return_sequences=n_sample)
            samples = [self.tokenizer.convert_ids_to_tokens(output_ids) for output_ids in output_ids_batch]       
    
        answers_batch = []
        for tokens in samples:
            tokens = tokens[1:] # ignore the first pad token
            nFiller = 0
            answers = []
            answer_start_index = 0
            for k, token in enumerate(tokens):
                if token == self.FILLER:
                    nFiller += 1
                    answers.append(self.tokenizer.convert_tokens_to_string(tokens[answer_start_index:k]))
                    answer_start_index = k + 1
                if nFiller == nBlank:
                    answers_batch.append(answers)
                    break        

        if len(answers_batch) < 1:
            for tokens in samples:
                print(self.tokenizer.convert_tokens_to_string(tokens))

        return answers_batch

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        return

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        return


class InfillingT5:
    def __init__(self, is_random_init=False, device='cuda'):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir='pretrained_models')
        if is_random_init:
            print('Initialize with random weights', file=sys.stderr)

            # using config from pretrained model
            config = AutoConfig.from_pretrained('t5-base', cache_dir='pretrained_models')

            self.model = T5ForConditionalGeneration(config).to(device)
            self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base', cache_dir='pretrained_models').to(device)

    def get_batch_loss(self, data_batch, device='cuda'):
        context_tokens_batch = [self.tokenizer.tokenize(' '.join(context)) + [self.tokenizer.eos_token] for context, _ in data_batch]
        answer_tokens_batch = [self.tokenizer.tokenize(' '.join(answer)) + [self.tokenizer.eos_token] for _, answer in data_batch]                

        context_ids_batch = [self.tokenizer.convert_tokens_to_ids(context_tokens) for context_tokens in context_tokens_batch]
        answer_ids_batch = [self.tokenizer.convert_tokens_to_ids(answer_tokens) for answer_tokens in answer_tokens_batch]

        batch_max_len = np.max([len(context_ids) for context_ids in context_ids_batch])
        context_ids_padded_batch = [context_ids + [self.tokenizer.eos_token_id for _ in range(batch_max_len - len(context_ids))] for context_ids in context_ids_batch]
        attention_mask = [[1 for _ in range(len(context_ids))] + [0 for _ in range(batch_max_len - len(context_ids))] for context_ids in context_ids_batch]
        attention_mask = torch.tensor(attention_mask).to(device)

        # List of decoder input/label lengths
        decoder_token_count_batch = [len(answer_tokens) for answer_tokens in answer_tokens_batch]

        decoder_batch_max_len = np.max(decoder_token_count_batch)
        decoder_label_ids_padded_batch = [answer_ids + [-100 for _ in range(decoder_batch_max_len - len(answer_ids))] for answer_ids in answer_ids_batch]
        decoder_attention_mask = [[1 for _ in range(len(answer_ids))] + [0 for _ in range(decoder_batch_max_len - len(answer_ids))] for answer_ids in answer_ids_batch]
        decoder_attention_mask = torch.tensor(decoder_attention_mask).to(device)

        input_ids = torch.tensor(context_ids_padded_batch).to(device)
        decoder_label_ids = torch.tensor(decoder_label_ids_padded_batch).to(device)

        output = self.model(input_ids, labels=decoder_label_ids, 
                        attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, return_dict=True)

        loss = output.loss

        batch_token_count = np.sum(decoder_token_count_batch)
        return loss, batch_token_count

    def get_loss(self, data, batch_size, device='cuda'):
        total_loss = 0
        total_token_count = 0

        for data_batch in get_batches(data, batch_size):
            loss, batch_token_count = self.get_batch_loss(data_batch, device=device)
            total_loss += loss.item()*batch_token_count
            total_token_count += batch_token_count

        return total_loss/total_token_count

    @staticmethod
    def count_blanks(tokens):
        blank_indice = []
        for i, token in enumerate(tokens):
            if token.startswith('<extra_id_'):
                blank_indice.append(i)
        nBlank = len(blank_indice) 
        return nBlank       

    def get_completion(self, sent, max_len=50, top_k=50, top_p=0.92, add_blank_symbol=True, device='cuda'):
        if add_blank_symbol:
            stimulus = sent
            context = add_indexed_masks(stimulus)
        else:
            context = sent
            stimulus = []
            for w in sent.split():
                if w.startswith('<extra_id_'):
                    stimulus.append('%%')
                else:
                    stimulus.append(w)
            stimulus = ' '.join(stimulus)

        context_tokens = self.tokenizer.tokenize(context) + [self.tokenizer.eos_token]
        context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        input_ids = torch.tensor(context_ids).unsqueeze(0).to(device)

        nBlank = self.count_blanks(context_tokens)

        output_ids = self.model.generate(input_ids, decoder_start_token_id=self.tokenizer.bos_token_id, do_sample=True, max_length=max_len, 
            top_k=top_k, top_p=top_p, temperature=1, num_return_sequences=1)
        samples = [self.tokenizer.convert_ids_to_tokens(sample) for sample in output_ids]        

        for tokens in samples:
            nFiller = 0
            for k, token in enumerate(tokens):
                if token.startswith('<extra_id_') or token.startswith('</s>'):
                    nFiller += 1
                if nFiller > nBlank:
                    print(formatted_response(stimulus, ['____' for _ in range(nBlank)])  + ' ' + self.tokenizer.convert_tokens_to_string(tokens[:(k+1)]))
                    break
        return

    def complete(self, stimulus, sampling=True, max_len=50, n_sample=100, top_k=50, top_p=0.92, temperature=1, num_beams=10, device='cuda'):
        """
        Assume the input stimulus uses %% as placeholders for blanks.
        Return generated completions to the incomplete stimulus.
        """
        context = add_indexed_masks(stimulus)
        context_tokens = self.tokenizer.tokenize(context) + [self.tokenizer.eos_token]
        context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        input_ids = torch.tensor(context_ids).unsqueeze(0).to(device)

        nBlank = self.count_blanks(context_tokens)   

        if not sampling:
            output_ids_batch = self.model.generate(input_ids, num_beams=num_beams, min_length=1, max_length=max_len, do_sample=sampling,
                                                    num_return_sequences=n_sample, early_stopping=True)
            samples = [self.tokenizer.convert_ids_to_tokens(output_ids) for output_ids in output_ids_batch]
        else:
            output_ids_batch = self.model.generate(input_ids, decoder_start_token_id=self.tokenizer.bos_token_id, max_length=max_len, do_sample=sampling,
                                                    top_k=top_k, top_p=top_p, temperature=temperature, num_return_sequences=n_sample)
            samples = [self.tokenizer.convert_ids_to_tokens(output_ids) for output_ids in output_ids_batch]     

        answers_batch = []
        for tokens in samples:
            last_mask_index = None
            answers = []
            nFiller = 0
            for k, token in enumerate(tokens):
                if token.startswith('<extra_id_') or token.startswith('</s>'):
                    nFiller += 1
                    if last_mask_index is not None:
                        answers.append(self.tokenizer.convert_tokens_to_string(tokens[(last_mask_index+1):k]))
                    last_mask_index = k
                if nFiller > nBlank:
                    answers_batch.append(answers)
                    break
        return answers_batch

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        return

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        return


class RerankingLM:
    def __init__(self, restore_from=None, device='cuda'):
        gpt_model_version = 'gpt2'
        # Load pre-trained GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_version, cache_dir='./pretrained_models')

        if restore_from is None:
            # Load pre-trained GPT-2 model
            self.model = GPT2LMHeadModel.from_pretrained(gpt_model_version, cache_dir='./pretrained_models').to(device)
            print('Load pretrained GPT-2 as reranking model', file=sys.stderr)
        else:
            # Random initialization
            config = GPT2Config(len(self.tokenizer))
            self.model = GPT2LMHeadModel(config).to(device)
            # Load trained model
            checkpoint = torch.load(restore_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print('Load reranking model params from {}'.format(restore_from), file=sys.stderr)         
        self.model.eval()


    def rerank_samples(self, samples, metric="avg_word_surprisal", detokenized=False):
        """
        Using pretrained language model to score and rerank samples.
        Use the argument detokenized to indicate whether the PTB tokenization scheme is detokenized or not.
        """
        samples_with_scores = []
        if detokenized:
            lines = samples
            tokenized_lines = [nltk.word_tokenize(line) for line in lines]
        else:
            lines = [TreebankWordDetokenizer().detokenize(line.strip().split()) for line in samples]
            tokenized_lines = samples
        for i, line in enumerate(lines):
            # rescore by GPT-2
            input_ids = self.tokenizer.encode(line, return_tensors='pt').cuda()
            outputs = self.model(input_ids, labels=input_ids)

            loss = outputs[0].item()
            
            samples_with_scores.append([i, line, loss, loss*input_ids.size()[1]/len(tokenized_lines[i]), loss*input_ids.size()[1]])
        
        if metric == "avg_word_surprisal":
            score_index = 3
        elif metric == "negllh":
            score_index = 4
        else:
            raise NotImplementedError

        sorted_samples_with_scores = sorted(samples_with_scores, key = lambda x: x[score_index])  # sorted according to averaged langauge modelling loss 
        return [(sorted_samples_with_scores[i][0], sorted_samples_with_scores[i][1], sorted_samples_with_scores[i][score_index]) for i in range(len(sorted_samples_with_scores))]
