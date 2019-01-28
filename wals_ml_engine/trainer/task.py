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

"""Job entry point for ML Engine."""

import argparse
import json
import os
import tensorflow as tf
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k

import model
import util
import wals


def train_and_evaluate_wals(args):
  # process input file
  input_file = util.ensure_local_file(args['train_files'][0])
  user_map, item_map, tr_sparse, test_sparse = model.create_test_and_train_sets(input_file)

  # train model
  output_row, output_col = model.train_model(args, tr_sparse)

  # save trained model to job directory
  model.save_model(args, user_map, item_map, output_row, output_col)

  # log results
  train_rmse = wals.get_rmse(output_row, output_col, tr_sparse)
  test_rmse = wals.get_rmse(output_row, output_col, test_sparse)

  precision, recall, pop_precision, pop_recall = wals.get_precision_recall(tr_sparse,
                                                                           test_sparse,
                                                                           output_row,
                                                                           output_col,
                                                                           5)

  if args['hypertune']:
    # write test_rmse metric for hyperparam tuning
    util.write_hptuning_metric(args, test_rmse)

  tf.logging.info('train RMSE = %.2f' % train_rmse)
  tf.logging.info('test RMSE = %.2f' % test_rmse)

  tf.logging.info('test precision = %.2f%%' % (100 * precision))
  tf.logging.info('test recall = %.2f%%' % (100 * recall))

  tf.logging.info('popularity precision = %.2f%%' % (100.0 * pop_precision))
  tf.logging.info('popularity recall = %.2f%%' % (100.0 * pop_recall))

def train_and_evaluate_lightfm(args):
  k = 5
  no_components = 11

  input_file = util.ensure_local_file(args['train_files'][0])

  print("Creating datasets ...")
  users, items, train, test = model.create_test_and_train_sets(input_file)

  print("Training model ...")
  lfm = LightFM(loss='warp', no_components=no_components, learning_schedule='adagrad')
  lfm.fit(train, epochs=20, num_threads=2)

  print("Evaluating model ...")
  test_precision = precision_at_k(lfm, test, train, k=k).mean()
  print("test precision@%d: %.2f%%" % (k, test_precision*100.0))
  test_recall = recall_at_k(lfm, test, train, k=k).mean()
  print("test recall@%d: %.2f%%" % (k, test_recall*100.0))

  train_precision = precision_at_k(lfm, train, k=k).mean()
  print("train precision@%d: %.2f%%" % (k, train_precision*100.0))
  train_recall = recall_at_k(lfm, train, k=k).mean()
  print("train recall@%d: %.2f%%" % (k, train_recall*100.0))

def parse_arguments():
  """Parse job arguments."""

  parser = argparse.ArgumentParser()
  # required input arguments
  parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # hyper params for model
  parser.add_argument(
      '--latent_factors',
      type=int,
      help='Number of latent factors',
  )
  parser.add_argument(
      '--num_iters',
      type=int,
      help='Number of iterations for alternating least squares factorization',
  )
  parser.add_argument(
      '--regularization',
      type=float,
      help='L2 regularization factor',
  )
  parser.add_argument(
      '--unobs_weight',
      type=float,
      help='Weight for unobserved values',
  )
  parser.add_argument(
      '--wt_type',
      type=int,
      help='Rating weight type (0=linear, 1=log)',
      default=wals.LINEAR_RATINGS
  )
  parser.add_argument(
      '--feature_wt_factor',
      type=float,
      help='Feature weight factor (linear ratings)',
  )
  parser.add_argument(
      '--feature_wt_exp',
      type=float,
      help='Feature weight exponent (log ratings)',
  )

  # other args
  parser.add_argument(
      '--output-dir',
      help='GCS location to write model, overriding job-dir',
  )
  parser.add_argument(
      '--verbose-logging',
      default=False,
      action='store_true',
      help='Switch to turn on or off verbose logging and warnings'
  )
  parser.add_argument(
      '--hypertune',
      default=False,
      action='store_true',
      help='Switch to turn on or off hyperparam tuning'
  )
  parser.add_argument(
      '--use-optimized',
      default=False,
      action='store_true',
      help='Use optimized hyperparameters'
  )

  args = parser.parse_args()
  arguments = args.__dict__

  # set job name as job directory name
  job_dir = args.job_dir
  job_dir = job_dir[:-1] if job_dir.endswith('/') else job_dir
  job_name = os.path.basename(job_dir)

  # set output directory for model
  if args.hypertune:
    # if tuning, join the trial number to the output path
    config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    trial = config.get('task', {}).get('trial', '')
    output_dir = os.path.join(job_dir, trial)
  elif args.output_dir:
    output_dir = args.output_dir
  else:
    output_dir = job_dir

  if args.verbose_logging:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  task_data = env.get('task') or {'type': 'master', 'index': 0}

  # update default params with any args provided to task
  params = model.DEFAULT_PARAMS
  params.update({k: arg for k, arg in arguments.iteritems() if arg is not None})
  if args.use_optimized:
    params.update(model.OPTIMIZED_PARAMS)

  params.update(task_data)
  params.update({'output_dir': output_dir})
  params.update({'job_name': job_name})
  params.update({'wt_type': wals.LOG_RATINGS})

  return params


if __name__ == '__main__':
  job_args = parse_arguments()
  train_and_evaluate_lightfm(job_args)


