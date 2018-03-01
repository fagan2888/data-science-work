import csv
import os

from sklearn.metrics import log_loss, roc_auc_score as AUC

preds, actuals = [1,0], [1,0]
cur_dir_path = os.path.dirname(os.path.realpath(__file__))

def score_prediction_csv(prediction_csv):
    with open(prediction_csv, 'r') as prediction_csv:
        csv_reader = csv.DictReader(prediction_csv)
        for row in csv_reader:
            preds.append(float(row['conversion_probability']))

    with open('{}/test_labels.txt'.format(cur_dir_path), 'r') as actuals_file:
        for line in actuals_file:
            actuals.append(float(line.split(' ')[1].rstrip()))

    print("AUC is: {}".format(AUC(actuals, preds)))
    print("log_loss is: {}".format(log_loss(actuals, preds)))
