import csv
import sys

from lib.scoring import score_prediction_csv
from model import fit_model_to_testing_data, get_fitted_model

HELP_MESSAGE = (
    "Usage: python train_and_score_model.py <train_data_tarfile> <test_data_tarfile> "
    "<csv_file_to_write_predictions_to>"
)


def main(train_data_tarfile, test_data_tarfile, predictions_csv_file):
    print("Getting fitted model...")
    fitted_model = get_fitted_model(train_data_tarfile)
    print("Done getting fitted model!")

    print("Using fitted model to predict testing event conversions...")
    ids, preds = fit_model_to_testing_data(fitted_model, test_data_tarfile)
    print("Done predicting event conversions!")

    print("Writing predictions as CSV file to {}...".format(predictions_csv_file))
    with open(predictions_csv_file, 'w') as csv_file:
        fieldnames = ['event_id', 'conversion_probability']
        csv_writer = csv.DictWriter(csv_file, fieldnames)
        csv_writer.writeheader()

        for event_id, pred in zip(ids, preds):
            csv_writer.writerow({'event_id': event_id, 'conversion_probability': pred})
    print("Done writing predictions file to {}!".format(predictions_csv_file))

    print("Using predictions file {} to score predictions...".format(predictions_csv_file))
    score_prediction_csv(predictions_csv_file)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(HELP_MESSAGE)
        sys.exit(1)

    train_data_tarfile = sys.argv[1]
    test_data_tarfile = sys.argv[2]
    predictions_csv_file = sys.argv[3]

    main(train_data_tarfile, test_data_tarfile, predictions_csv_file)

