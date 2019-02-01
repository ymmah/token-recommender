# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WALS model input data, training and predict functions."""

import os
import pickle

import numpy as np
import pandas as pd
import sh
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from pandas import Series
from scipy.sparse import coo_matrix
from tqdm import tqdm

# ratio of train set size to test set size
TEST_SET_RATIO = 10

# parameters optimized with hypertuning
OPTIMIZED_PARAMS = {
    'num_iters': 20
}


def create_test_and_train_sets(input_file):
    return _token_balances_train_and_test(input_file)


def _token_balances_train_and_test(input_file):
    headers = ['token_address', 'user_address', 'rating']
    balances_df = pd.read_csv(input_file,
                              sep=',',
                              names=headers,
                              header=0,
                              dtype={
                                  'token_address': np.str,
                                  'user_address': np.str,
                                  'rating': np.float32,
                              })

    df_tokens = pd.DataFrame({'token_address': balances_df.token_address.unique()})
    df_sorted_tokens = df_tokens.sort_values('token_address').reset_index()
    pds_tokens = df_sorted_tokens.token_address

    # preprocess data. df.groupby.agg sorts user_address and token_address
    df_user_token_addresses = balances_df.groupby(['user_address', 'token_address']).agg({'rating': 'sum'})

    # create a list of (user_address, token_address, rating) records, where user_address and
    # token_address are 0-indexed
    current_user = -1
    user_index = -1
    ratings = []
    users = []
    for user_token_rating in df_user_token_addresses.itertuples():
        user = user_token_rating[0][0]
        token = user_token_rating[0][1]

        # as we go, build a (sorted) list of user addresses
        if user != current_user:
            users.append(user)
            user_index += 1
            current_user = user

        # this search makes the preprocessing time O(r * i log(i)),
        # r = # ratings, i = # tokens
        token_index = pds_tokens.searchsorted(token)[0]
        ratings.append((user_index, token_index, user_token_rating[1]))

    # convert ratings list and user list to np array
    ratings = np.asarray(ratings)
    users = np.asarray(users)

    # create train and test sets
    train_sparse, test_sparse = _create_sparse_train_and_test(ratings,
                                                              user_index + 1,
                                                              df_tokens.size)

    return users, pds_tokens.as_matrix(), train_sparse, test_sparse


def _create_sparse_train_and_test(ratings, n_users, n_items):
    # pick a random test set of entries, sorted ascending
    test_set_size = len(ratings) // TEST_SET_RATIO
    np.random.seed(42)
    test_set_idx = np.random.choice(range(len(ratings)), size=test_set_size, replace=False)
    test_set_idx = sorted(test_set_idx)

    # sift ratings into train and test sets
    ts_ratings = ratings[test_set_idx]
    tr_ratings = np.delete(ratings, test_set_idx, axis=0)

    # create training and test matrices as coo_matrix's
    u_tr, i_tr, r_tr = zip(*tr_ratings)
    tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))

    u_ts, i_ts, r_ts = zip(*ts_ratings)
    test_sparse = coo_matrix((r_ts, (u_ts, i_ts)), shape=(n_users, n_items))

    return tr_sparse, test_sparse


def train_model(args, train):
    no_components = 11
    model = LightFM(loss='warp', no_components=no_components, learning_schedule='adagrad')
    model.fit(train, epochs=args['num_iters'], num_threads=2)
    return model


def save_model(args, model, users, items):
    model_dir = os.path.join(args['output_dir'], 'model')

    # if our output directory is a GCS bucket, write model files to /tmp,
    # then copy to GCS
    gs_model_dir = None
    if model_dir.startswith('gs://'):
        gs_model_dir = model_dir
        model_dir = '/tmp/{0}'.format(args['job_name'])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'model.pickle'), 'wb') as output_file:
        pickle.dump(model, output_file)
    np.save(os.path.join(model_dir, 'user'), users)
    np.save(os.path.join(model_dir, 'item'), items)

    if gs_model_dir:
        sh.gsutil('cp', '-r', os.path.join(model_dir, '*'), gs_model_dir)


def evaluate_model(args, mdl, train, test):
    k = 5
    test_precision = precision_at_k(mdl, test, train, k=k).mean()
    print("test precision@%d: %.2f%%" % (k, test_precision * 100.0))
    test_recall = recall_at_k(mdl, test, train, k=k).mean()
    print("test recall@%d: %.2f%%" % (k, test_recall * 100.0))

    train_precision = precision_at_k(mdl, train, k=k).mean()
    print("train precision@%d: %.2f%%" % (k, train_precision * 100.0))
    train_recall = recall_at_k(mdl, train, k=k).mean()
    print("train recall@%d: %.2f%%" % (k, train_recall * 100.0))

    if args.get('eval_popularity') == 'True':
        pop_precision = precision_at_k(popularity_model(k=k), test, train, k=k).mean()
        print("popularity precision@%d: %.2f%%" % (k, pop_precision * 100.0))
        pop_recall = recall_at_k(popularity_model(k=k), test, train, k=k).mean()
        print("popularity recall@%d: %.2f%%" % (k, pop_recall * 100.0))


def popularity_model(k=10):
    def get_most_popular(sparse):
        """Get the k most popular items in sparse based on number of ratings."""
        return list(Series(sparse.nonzero()[1]).value_counts()[:(k * 10)].index)

    class PopularityModel:
        def predict_rank(self, test_interactions, train_interactions, **kwargs):
            most_popular = get_most_popular(test_interactions)
            num_users = test_interactions.shape[0]
            data = []
            rows = []
            cols = []
            for user in tqdm(range(0, num_users)):
                test_items = set(test_interactions.getrow(user).indices)
                train_items = set(train_interactions.getrow(user).indices)
                recommended_items = [item for item in most_popular if item not in train_items]
                predictions = set(recommended_items).intersection(test_items)

                recommended_ranks = {item: index for index, item in enumerate(recommended_items)}

                for prediction in predictions:
                    rank = recommended_ranks.get(prediction)
                    if rank is not None and rank < k:
                        data.append(float(rank))
                        rows.append(user)
                        cols.append(prediction)

            ranks = sp.csr_matrix((data, (rows, cols)), shape=test_interactions.shape)

            return ranks

    return PopularityModel()


def generate_recommendations(user_idx, user_rated, model, k, item_count):
    k_r = k + len(user_rated)
    scores = model.predict(user_idx, np.arange(item_count))
    top_items = np.argsort(-scores)
    candidate_items = top_items[:k_r]

    # remove previously rated items and take top k
    recommended_items = [i for i in candidate_items if i not in user_rated]
    recommended_items = recommended_items[:k]

    return recommended_items
