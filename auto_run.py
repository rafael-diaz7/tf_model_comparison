import subprocess

subprocess.run(['mkdir', 'grid_search'])
subprocess.run(['mkdir', 'models'])

learnable = ['True', 'False']
architecture = ['CLS_1L', 'CLS_3L', 'biLSTM_3L', 'biLSTM_1L', 'BERT_SIG']
learning_rates = [str((1 * (10 ** -x))) for x in range(8, 0, -1)] # 1 * 10^x ... 1e-8 -> 1e-1
drop_out = ['0', '0.5', '0.8']

# HYPERPARAMETER SEARCH

# BIO DISCHARGE SUMMARY BERT model
# Learning Rates: [1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1] 8
# Dropout Rates: [0, 0.5, 0.8] 3
# Back Propogation: [False, True] 2
# Architectures: [3L CLS, 1L CLS, 3L biLSTM, 1L biLSTM, Bert->sigmoid]  5
# 240 different combinations

for i in learnable:
    for j in architecture:
        for k in learning_rates:
            for l in drop_out:
                subprocess.run(['nohup', 'python3', 'experiment.py', k, l, i, j])
