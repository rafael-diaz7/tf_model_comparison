import subprocess

subprocess.run(['mkdir', 'metrics'])

learning_rates = [str((1 * (10 ** -x))) for x in range(8, 1, -1)] # 1 * 10^x ... 1e-8 -> 1e-1
drop_out = ['0', '0.5', '0.9']

# HYPERPARAMETER SEARCH

# BIO DISCHARGE SUMMARY BERT model
# Learning Rates: [1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1]
# Dropout Rates: [0, 0.5, 0.9]
# 24 different combinations

for i in learning_rates:
    for j in drop_out:
        subprocess.run(['nohup', 'python3', 'experiment.py', i, j])