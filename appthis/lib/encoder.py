from datetime import datetime

import numpy as np

from lib.feature_defs import (
    context_cat_placement, context_cat_user, context_num_placement, context_num_user,
    offer_cat, offer_num,
)


class VectorEncoder():

    def __init__(self):
        self.D = 2 ** 14 # ~16k

    def _get_time_features(self, req):
        time = req['createdOn'].split('T')
        day = time[0]
        features = []
        features.append(str(datetime.strptime(day, '%Y-%m-%d').weekday())), #day of week
        features.append(time[1].split(':')[0]) #hour of day
        features.append(str(datetime.strptime(day, '%Y-%m-%d').day)) #day of month
        features = [f for f in map(self.hash, features)]

        return features

    def _make_offer_features(self, offer):
        Xo_cat, Xo_num = [], []
        for f in offer_cat:
            Xo_cat.append(self.hash(offer.get('attributes', {}).get(f, "0")))
        for f in offer_num:
            v = offer.get('attributes', {}).get(f, "0")
            try:
                Xo_num.append(float(v))
            except ValueError:
                Xo_num.append(0.0)
        return Xo_cat, Xo_num

    def _make_context_features(self, req, X_cat):
        X_num = []
        for f in context_cat_placement:
            X_cat.append(self.hash(req['placement']['attributes'].get(f, "0")))
        for f in context_cat_user:
            X_cat.append(self.hash(req['user']['attributes'].get(f, "0")))
        for f in context_num_placement:
            v = req['placement']['attributes'].get(f, "0")
            try:
                X_num.append(float(v))
            except ValueError:
                X_num.append(0.0)
        for f in context_num_user:
            v = req['user']['attributes'].get(f, "0")
            try:
                X_num.append(float(v))
            except ValueError:
                X_num.append(0.0)
        return X_cat, X_num

    def normed_modulus(self, int):
        return float(int % self.D) / self.D

    def hash(self, str):
        val = hash(str)
        return self.normed_modulus(val)

    def encode(self, req):
        req = req['event']

        X_time = self._get_time_features(req)
        X_cat, X_num = self._make_context_features(req, X_time)

        Xs = []
        for offer in req.get('offers'):
            Xo_cat, Xo_num = self._make_offer_features(offer)
            Xs.append(np.concatenate((X_num, Xo_num, X_cat, Xo_cat)))
        return Xs

