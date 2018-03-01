import sys
import pandas as pd

from lib.data import data_iterator
from lib.encoder import VectorEncoder

data_tarfile = sys.argv[1]
data_output = sys.argv[2]

Xs, ys = [], []
vect_encoder = VectorEncoder()

# This block simply goes through the training data tarball and appends event instances to the
# `Xs` list and appends the labels (or "conversions") for those event instances to the `ys`
# list.
print("Transforming data from {}...".format(data_tarfile))

in_data_iterator = data_iterator(data_tarfile)
for json_obj in in_data_iterator:
    ys.append(json_obj['event']['conversion'])
    Xs.append(vect_encoder.encode(json_obj)[0])

df = pd.DataFrame(Xs)
df.to_csv(data_output)

print("Done transforming data!")