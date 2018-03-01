import os
import pandas as pd
import numpy as np

DATA_DIR = '/home/jeff/Downloads/data/kkbox'

def get_train():
    return pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), index_col=0)

def get_train_Xy():
    members = get_members()
    trans_derived = get_trans_derived()

    train = get_train()
    df = train.join(members, how='left')
    df = df.join(trans_derived, how='left')
    df.fillna(method='pad', inplace=True)

    return df




def get_test():
    return pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), index_col=0)

def get_members(manipulate=True):
    members = pd.read_csv(os.path.join(DATA_DIR, 'members_filtered.csv'), index_col=0)

    if manipulate:
        # replace registration/expiration dates with account life
        members['account_life'] = members['expiration_date'] - members['registration_init_time']
        members.drop(['registration_init_time', 'expiration_date'], axis=1, inplace=True)

        # replace "bd" with age and give everyone with improper ages a mean value
        sensible_values = members['bd'].between(10,90)
        members['bd'][~sensible_values] = members['bd'][sensible_values].median()

        members.fillna('backfill', inplace=True)
        members = pd.get_dummies(members, columns=['city', 'registered_via', 'gender'])

    return members

def get_transactions(manipulate=True):
    transactions = pd.read_csv(os.path.join(DATA_DIR, 'transactions_filtered.csv'), index_col=0)

    return transactions

def get_trans_derived():
    transactions = get_transactions()

    trans_derived = pd.DataFrame(index=transactions.index)
    trans_derived[['mean_cancel', 'num_orders']] = transactions.groupby(level=0)['is_cancel'].agg(['mean', 'count'])
    
    return trans_derived