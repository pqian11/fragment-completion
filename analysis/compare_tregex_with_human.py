import numpy as np
import pandas as pd


def get_precision_recall_acc(human_judgments, tregex_judgments):
    n_item = len(human_judgments)

    tp = 0
    tn = 0

    for i in range(n_item):
        if human_judgments[i] == 1 and tregex_judgments[i] == 1:
            tp += 1
        if human_judgments[i] == 0 and tregex_judgments[i] == 0:
            tn += 1
    return tp/np.sum(tregex_judgments), tp/np.sum(human_judgments), (tp+tn)/n_item  


def print_latex_table_source(df):
    print('{:<3} & {:<30} & {:<10}  &  {:<10}  & {:<10}\\\\\\midrule'.format('Test Index', 'Test Name', 'Precision', 'Recall', 'Accuracy'))
    for test_index, test_name in enumerate(test_names):
        human_judgments = df[df['group'] == test_name]['human_judgment'].to_list()
        tregex_judgments = df[df['group'] == test_name]['tregex_judgment'].to_list()    
        precision, recall, acc = get_precision_recall_acc(human_judgments, tregex_judgments)
        print('({}) & {:<30} & {:<10.3f}  &  {:<10.3f}  & {:<10.3f}\\\\'.format(test_index+1, test_name2full_name[test_name], precision, recall, acc))      


df = pd.read_csv('data/exp1/compare_tregex_with_human_judgments.csv')

human_judgments = df['human_judgment'].to_list()
tregex_judgments = df['tregex_judgment'].to_list()

n_item = len(human_judgments)


tp = 0

for i in range(n_item):
    if human_judgments[i] == 1 and tregex_judgments[i] == 1:
        tp += 1

print('Precision: {}'.format(tp/np.sum(tregex_judgments)))
print('Recall: {}'.format(tp/np.sum(human_judgments)))


models = ['human', 'gibbscomplete', 't5-finetune', 'bart-finetune', 'ilm']

print('{:<15} {:<10}   {:<10}   {:<10}'.format('model', 'precision', 'recall', 'acc'))
for model in models:
    human_judgments = df[df['model'] == model]['human_judgment'].to_list()
    tregex_judgments = df[df['model'] == model]['tregex_judgment'].to_list()    
    precision, recall, acc = get_precision_recall_acc(human_judgments, tregex_judgments)
    print('{:<15} {:<10.3}   {:<10.3}   {:<10.3}'.format(model, precision, recall, acc))



test_names = ["agreement_subj", "agreement_subj-long", "agreement_emb-subj-long", "agreement_subj-with-coord", "agreement_subj-with-PP",
                "clause_VP","clause_VP-with-PP-adjunct", "clause_VP-with-adjunct-long",
                "clause_VP-with-complement", "clause_VP-with-complement-long", "clause_VP-gerund", 
                "clause_phrasal-verb", "clause_phrasal-verb-with-subj", 
                "clause_resultative", "clause_resultative-long",
                "coord_S", "coord_VP", "coord_emb-NP", "coord_emb-VP",
                "coord_either", "coord_neither", "coord_gap-NP", "gap_adjunct", "gap_obj", "gap_subj", "gap_phrasal-verb"]

test_full_names = [
    "Number Agreement", "Number Agreement (Long Subject)", "Number Agreement (Embedded Clause)", "Number Agreement (Coordination)", "Number Agreement (with PP)",
    "Clausal Structure","Clausal Structure (PP Adjunct)", "Clausal Structure (Long Adjunct)",
    "Clausal Structure (Complement)", "Clausal Structure (Long Complement)", "Gerund", 
    "Phrasal Verb", "Phrasal Verb (with NP)", 
    "Resultative", "Resultative (Long NP)",
    "S Coordiation", "VP Coordination", "Embedded NP Coordination", "Embedded VP Coordination",
    "Coordination (either)", "Coordination (neither)", "Coordination in wh-clause", "Filler-Gap (Adjunct)", "Filler-Gap (Object)", "Filler-Gap (Subject)", "Filler-Gap (Phrasal Verb)"
]

test_name2full_name = dict(zip(test_names, test_full_names))

print('{:<30} {:<10}   {:<10}   {:<10}'.format('test_name', 'precision', 'recall', 'acc'))
for test_name in test_names:
    human_judgments = df[df['group'] == test_name]['human_judgment'].to_list()
    tregex_judgments = df[df['group'] == test_name]['tregex_judgment'].to_list()    
    precision, recall, acc = get_precision_recall_acc(human_judgments, tregex_judgments)
    print('{:<30} {:<10.3}   {:<10.3}   {:<10.3}'.format(test_name, precision, recall, acc))    


# print_latex_table_source(df)
