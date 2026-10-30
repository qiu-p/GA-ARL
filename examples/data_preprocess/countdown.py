# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess countdown dataset.

"""

import argparse
import os
import re

import datasets
from string import Template

from verl.utils.hdfs_io import copy, makedirs

def make_prefix(dp):
    target = dp['target']
    numbers = dp['nums']
    prefix = f"""
    In this task, you need to use a list of numbers x = {numbers} to create an equation that leads up to the target number y = {target} using the basic arithmetic operations (+, -, *, /), and each number can only be used once. Think and return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
    """
    return prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/countdown", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "Jiayi-Pan/Countdown-Tasks-3to4"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    dataset = dataset['train'].train_test_split(train_size=10000, test_size=1000, seed=123)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example)
            solution = {
                "target": example['target'],
                "numbers": example['nums']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example['target']
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'solution': solution
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    print(train_dataset[0])
    
    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
