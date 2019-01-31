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

import model
import util


def train_and_evaluate_lightfm(args):
    input_file = util.ensure_local_file(args['train_files'][0])

    print("Creating datasets ...")
    users, items, train, test = model.create_test_and_train_sets(input_file)

    print("Training model ...")
    mdl = model.train_model(args, train)

    print("Saving model ...")
    model.save_model(args, mdl, users, items)

    print("Evaluating model ...")
    model.evaluate_model(args, mdl, train, test)


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
    params = model.OPTIMIZED_PARAMS
    params.update({k: arg for k, arg in arguments.items() if arg is not None})

    params.update(task_data)
    params.update({'output_dir': output_dir})
    params.update({'job_name': job_name})

    return params


if __name__ == '__main__':
    job_args = parse_arguments()
    train_and_evaluate_lightfm(job_args)
