# Flexible Generation from Fragmentary Linguistic Input

This repository accompanies the paper, Flexible Generation from Fragmentary Linguistic Input, to appear at ACL 2022. It includes the implementation of fine-tuning pretrained models or training models from scratch on a text infilling task, as well as GibbsComplete, an algorithm that combines masked and autpregressive language models to fill in the blanks in a sentence. It also includes the experimental materials used in the evaluations in the paper.

## Environment

The Singularity definition file `scripts/py36-torch-transformers3.2.0.def` can be used to build the container for running the code for model training and inference.

## GibbsComplete

The script `src/gibbs_complete.py` implement the algorithm. To fill in the blanks of sentences with pretrained masked language model and autoregressive language model:

```
python src/gibbs_complete.py --fpath $INPUT --n_sample 1000 --batch_size 100 --rerank
```

Train masked language model component from scratch for GibbsComplete:

```
python src/masked_lm.py --do_train --lr ${LR} --batch_size ${BATCH_SIZE} --epoch ${N_EPOCHS} --train_data ${TRAIN_DATA} --dev_data ${DEV_DATA} --save_path ${MODEL_PATH} --report 1000 --sample_every 10000
```

To fill in the blanks with masked language model component and forward language model component learned from scratch:

```
python src/gibbs_complete.py --fpath $INPUT --n_sample 1000 --batch_size 100 --output_path ${OUTPUT_ANSWERS_PATH} --restore_from ${MASKED_LM_PARAMS} --restore_rerank_lm_from ${FORWARD_LM_PARAMS} --rerank
```

## ILM, InfillT5, InfillBART
To fine-tune pretrained models on sentence infilling task:

```
python src/infill.py --seed ${SEED} --model_type ${MODEL_TYPE} --lr ${LR} --do_train --batch_size ${BATCH_SIZE} --epochs ${N_EPOCHS} --save_path ${MODEL_PATH} --train_data ${TRAIN_DATA} --dev_data ${DEV_DATA} --report 1000 --sample_every 5000
```

To train models from scratch on the sentence infilling task:

```
python src/infill.py --seed ${SEED} --model_type ${MODEL_TYPE} --lr ${LR} --do_train --batch_size ${BATCH_SIZE} --epochs ${N_EPOCHS} --save_path ${MODEL_PATH} --train_data ${TRAIN_DATA} --dev_data ${DEV_DATA} --report 1000 --sample_every 5000 --random_init
```

The following command uses trained models to generate completions. The commandline argument can take the value, `gpt2`, `t5`, and `bart`, which correspond to ILM, InfillT5, InfillBART respectively.

```
python src/infill.py --model_type ${MODEL_TYPE} --restore_from ${MODEL_PARAMS} --fpath $INPUT --max_len 50 --do_test --n_sample 35 --n_output 35
```

## Plot figures

The `analysis` folder contains the code and data for generating the figures in t
he paper. The following commands run the plotting scripts and generate figures i
n the `figs` folder.

```
cd analysis
mkdir -p fig/exp0
mkdir -p fig/exp1
mkdir -p fig/exp2

# Plot the figure for Evaluation I
python exp0_analysis.py

# Plot the figures for Evaluation II
python exp1_analysis.py
python exp1_analysis.py --rerank

# Plot the figures for Evaluation III
python exp2_analysis.py
python exp2_analysis.py --rerank
```

