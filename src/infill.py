from transformers import GPT2Tokenizer, GPT2LMHeadModel, BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, AdamW
import torch
import argparse
import random
import numpy as np
import sys
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from helper import *
from InfillingModels import *
import utils
import functools
print = functools.partial(print, flush=True)


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default=None, help='Specific type of infilling models.')
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
parser.add_argument('--save_path', type=str, default='model.params', help='Path to saving model checkpoint.')
parser.add_argument('--batch_size', type=int, default=1, help="Size of a training batch.")
parser.add_argument('--random_init', action='store_true', help='Random initialization of model parameters.')
parser.add_argument('--fpath', type=str, default=None, help='File path to a list of incomplete sentences.')
parser.add_argument('--rerank', action='store_true', help='Whether to rerank completions')
parser.add_argument('--filter_repeat', action='store_true', help='Whether to filter repeated completions')
parser.add_argument('--restore_rerank_lm_from', type=str, default=None, help='Path to trained reranking LM model.')
parser.add_argument('--max_len', type=int, default=50, help='Max length for sampling filling content of the blanks.')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling completions.')
parser.add_argument('--top_k', type=int, default=50, help='Sample from top k.')
parser.add_argument('--n_sample', type=int, default=25, help='Number of samples')
parser.add_argument('--n_output', type=int, default=None, help='Number of output if reranking the candidate completions.')
parser.add_argument('--early_stopping_threshold', type=int, default=2, help='Threshold for early stopping during training.')
parser.add_argument('--output_path', type=str, default=None, help="Path to save the output answers.")

args = parser.parse_args()


def has_empty_answer(answers):
    for ans in answers:
        if ans.strip() == '':
            return True
    return False


def infill(infilling_model, stimulus, rerank_lm=None, max_len=50, temperature=1, top_k=50, n_sample=50, sample_batch_size=50, n_output=None, do_rerank=False, rerank_metric='negllh', filter_repeat=False):
    """
    n_sample is the total number of samples before reranking
    n_output is the number of output completion.
    """
    # print(stimulus)

    # Add space between the final period and the last blank.
    stimulus = adjust_stimulus(stimulus)

    if sample_batch_size == None or (sample_batch_size > n_sample):
        sample_batch_size = n_sample

    if not filter_repeat:
        n_batch = n_sample//sample_batch_size if (n_sample % sample_batch_size == 0) else (n_sample//sample_batch_size + 1)
        answers_all = []

        while len(answers_all) < n_sample:
            answers_batch = infilling_model.complete(stimulus, sampling=True, max_len=max_len, temperature=temperature, top_k=top_k, n_sample=sample_batch_size)
            answers_all += [answers for answers in answers_batch if not has_empty_answer(answers)]
        answers_all = answers_all[:n_sample]
    else:
        completion_set = set([])
        answers_all = []
        while len(completion_set) < n_sample:
            answers_batch = infilling_model.complete(stimulus, sampling=True, max_len=50, temperature=temperature, top_k=top_k, n_sample=sample_batch_size)
            for answers in answers_batch:
                completion = formatted_response(stimulus, answers)
                if completion not in completion_set:
                    if not has_empty_answer(answers):
                        completion_set.add(completion)
                        answers_all.append(answers)
        answers_all = answers_all[:n_sample]

    if do_rerank:
        assert rerank_lm is not None
        assert n_output is not None
        completions = [formatted_response(stimulus, answers) for answers in answers_all]
        reranked_completions_with_scores = rerank_lm.rerank_samples([sent for sent in completions], metric=rerank_metric, detokenized=True)
        return [answers_all[index] for index, completion, score in reranked_completions_with_scores[:n_output]]
    else:
        return answers_all


def generate_dev_data(infilling_model, dev_lines, seed=0):
    rng = np.random.RandomState(seed=seed)
    if MODEL_TYPE == 't5':
        dev_data = [add_gaps(tokens, special_format='T5', rng=rng) for tokens in dev_lines]
    else:
        dev_data = [add_gaps(tokens, BLANK=infilling_model.BLANK, FILLER=infilling_model.FILLER, rng=rng) for tokens in dev_lines]
    return dev_data

RANDOM_SEED = args.seed if args.seed is not None else int(np.random.random()*10000)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = torch.nn.CrossEntropyLoss()

MODEL_TYPE = args.model_type

if MODEL_TYPE  == 'gpt2':
    infilling_model = InfillingLM(is_random_init=args.random_init, device=device)
elif MODEL_TYPE  == 'bart':
    infilling_model = InfillingBART(is_random_init=args.random_init, device=device)
elif MODEL_TYPE  == 't5':
    infilling_model = InfillingT5(is_random_init=args.random_init, device=device)
else:
    raise NotImplementedError


if args.restore_from is not None:
    checkpoint = torch.load(args.restore_from)
    infilling_model.model.load_state_dict(checkpoint["model_state_dict"])
    print("Load from: {}".format(args.restore_from), file=sys.stderr)


if args.do_train:
    # Path to save the newly trained model
    CHECKPOINT_PATH = args.save_path

    # Print out training settings
    print('Training batch size: {}'.format(args.batch_size), file=sys.stderr)
    print('Learning rate: {}'.format(args.lr), file=sys.stderr)
    print('Model path: {}'.format(CHECKPOINT_PATH), file=sys.stderr)

    # Set the learning rate of the optimizer
    optimizer = AdamW(infilling_model.model.parameters(), lr=args.lr)

    # Load train and dev data
    train_lines = load_data(args.train_data)
    dev_lines = load_data(args.dev_data)

    dev_data = generate_dev_data(infilling_model, dev_lines, seed=0)

    if args.restore_from:
        infilling_model.model.eval()
        best_validation_loss = infilling_model.get_loss(dev_data, batch_size=args.batch_size)
        infilling_model.model.train()
        print('resume training; validation loss: {}'.format(best_validation_loss))
    else:
        best_validation_loss = np.Inf

    starting_epoch = checkpoint['epoch'] + 1 if (args.restore_from is not None) else 0
    n_epochs = args.epochs
    no_improvement_count = checkpoint['no_improvement_count'] if (args.restore_from is not None) else 0
    VALID_EVERY = None if ((args.valid_every is None) or (args.valid_every < 1)) else args.valid_every

    early_stopping_counter = utils.EarlyStopping(best_validation_loss=best_validation_loss, no_improvement_count=no_improvement_count, threshold=args.early_stopping_threshold)

    for epoch in range(starting_epoch, n_epochs):
        np.random.shuffle(train_lines)

        count = 0  # cumulative count of training examples
        batch_count = 0 # cumulative count of training batches

        for line_batch in get_batches(train_lines, args.batch_size):
            optimizer.zero_grad()

            # Generate training data by randomly cropping spans of words
            if MODEL_TYPE == 't5':
                train_data_batch = [add_gaps(tokens, special_format='T5') for tokens in line_batch]
            else:
                train_data_batch = [add_gaps(tokens, BLANK=infilling_model.BLANK, FILLER=infilling_model.FILLER) for tokens in line_batch]

            loss, batch_token_count = infilling_model.get_batch_loss(train_data_batch)
            loss.backward()
            optimizer.step()

            count += len(train_data_batch)
            batch_count += 1       

            if batch_count > 0 and batch_count % args.report == 0:
                print('Epoch {:.3f} loss: {}'.format(epoch + count/len(train_lines), loss.item()))

            if batch_count > 0 and batch_count % args.sample_every == 0:
                infilling_model.model.eval()
                test_sentences = ['{blank} published won {blank} .', 
                                    'The {blank} published by a local press that just formed last year has won {blank} .', 
                                    '{blank} what the {blank} book.',
                                    ]
                with torch.no_grad():
                    for test_sentence in test_sentences:
                        infilling_model.get_completion(test_sentence.format(blank='%%'), add_blank_symbol=True)
                    infilling_model.get_completion(' '.join(train_data_batch[0][0]), add_blank_symbol=False)

                infilling_model.model.train()

            if VALID_EVERY is not None:
                if batch_count > 0  and batch_count % VALID_EVERY == 0:
                    infilling_model.model.eval()
                    with torch.no_grad():
                        validation_loss = infilling_model.get_loss(dev_data, batch_size=args.batch_size)
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
                            'model_state_dict': infilling_model.model.state_dict(),
                            'loss': validation_loss},
                            CHECKPOINT_PATH)

                    infilling_model.model.train()


        # Validation after each epoch
        infilling_model.model.eval()

        with torch.no_grad():
            validation_loss = infilling_model.get_loss(dev_data, batch_size=args.batch_size)
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
                'model_state_dict': infilling_model.model.state_dict(),
                'loss': validation_loss},
                CHECKPOINT_PATH)
            
        infilling_model.model.train()


if args.do_test:
    def adjust_stimulus(stimulus):
        if stimulus[-3:-1] == '%%':
            return stimulus[:-1] + ' ' + stimulus[-1]
        else:
            return stimulus

    if args.fpath is not None:
        with open(args.fpath) as f:
            stimuli = f.readlines()
        stimuli = [stimulus.strip() for stimulus in stimuli if stimulus.strip() != '']
    else:
        # use demo examples
        stimuli = ["%% published won %% .", 
                    "The %% published by a local press that just formed last year has won %% .",
                    "%% what the %% book.",
                    "%% difficult %% impossible %% .",
                    "He is one of %% jazz on a violin.",
                    "%% museum %% city %% ."]

    TEMPERATURE = args.temperature
    TOP_K = args.top_k
    N_SAMPLE = args.n_sample
    N_OUTPUT = args.n_output
    MAX_LEN = args.max_len
    FILTER_REPEAT = args.filter_repeat
    DO_RERANK = args.rerank

    OUTPUT_PATH = args.output_path if (args.output_path is not None) else None

    if DO_RERANK:
        # Load reranking language model
        rerank_lm = RerankingLM(restore_from=args.restore_rerank_lm_from, device=device)
    else:
        rerank_lm=None

    print("Infilling max length: {}\nSampling temperature: {}\nTop-k: {}\nNumer of samples: {}\nFilter repeat: {}".format(MAX_LEN, TEMPERATURE, TOP_K, N_SAMPLE, FILTER_REPEAT), file=sys.stderr)
    print("Rerank: {}\nNumber of reranking output: {}".format(DO_RERANK, N_OUTPUT), file=sys.stderr)

    infilling_model.model.eval()

    if OUTPUT_PATH is not None:
        outf = open(OUTPUT_PATH, 'w')

    with torch.no_grad():
        for stimulus in stimuli:
            answers_all = infill(infilling_model, stimulus, rerank_lm=rerank_lm, max_len=MAX_LEN, temperature=TEMPERATURE, top_k=TOP_K, n_sample=N_SAMPLE, n_output=N_OUTPUT, do_rerank=DO_RERANK, filter_repeat=FILTER_REPEAT, rerank_metric='avg_word_surprisal')
            
            print(stimulus)
            for answers in answers_all:
                print(formatted_response(stimulus, answers))
            print()

            if OUTPUT_PATH is not None:
                outf.write(stimulus+'\n')
                outf.writelines(['\t'.join(answers)+'\n' for answers in answers_all])
                outf.write('\n')
                outf.flush()

    if OUTPUT_PATH is not None:
        outf.close()

    print("Successfully finished!", file=sys.stderr)
