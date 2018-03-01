import numpy
import os
import data_manip

def get_idx():
    train = data_manip.get_train()
    test = data_manip.get_test()
    idx = train.index.union(test.index)

    return idx

def reduce_members():
    idx = get_idx()

    members = data_manip.get_members(False)
    members = members.filter(idx, axis=0)
    members.to_csv(os.path.join(data_manip.DATA_DIR, 'members_filtered.csv'))

def reduce_transactions():
    idx = get_idx()

    transactions = data_manip.get_transactions(False)
    transactions = transactions.loc[idx,:]
    transactions.to_csv(os.path.join(data_manip.DATA_DIR, 'transactions_filtered.csv'))


