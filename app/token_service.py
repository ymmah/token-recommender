import logging
import os

import google.auth
import google.cloud.storage as storage
import pandas as pd

logging.basicConfig(level=logging.INFO)

LOCAL_PATH = '/tmp'

TOKENS_FILE = 'data/tokens.csv'


class TokenService(object):

    def __init__(self, local_model_path=LOCAL_PATH):
        _, project_id = google.auth.default()
        self._bucket = 'recserve_' + project_id
        self._load_tokens(local_model_path)

    def _load_tokens(self, local_model_path):
        # download files from GCS to local storage
        os.makedirs(os.path.join(local_model_path, 'data'), exist_ok=True)
        client = storage.Client()
        bucket = client.get_bucket(self._bucket)

        logging.info('Downloading blobs for TokenService.')

        blob = bucket.blob(TOKENS_FILE)
        with open(os.path.join(local_model_path, TOKENS_FILE), 'wb') as file_obj:
            blob.download_to_file(file_obj)

        logging.info('Finished downloading blobs for TokenService.')

        # load tokens
        tokens_df = pd.read_csv(os.path.join(local_model_path, TOKENS_FILE), sep=',', header=0)
        tokens_df['address_index'] = tokens_df['address']
        tokens_df.set_index('address_index', inplace=True)
        self.tokens = tokens_df.to_dict('index')

    def get_token_by_address(self, address):
        token = self.tokens.get(address)
        return token.copy() if token is not None else None
