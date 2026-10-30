# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""Generation batches and bounded replay storage."""
import gc
import math
from collections import deque
import numpy as np
import torch
from verl import DataProto
from ..utils import OnlinePolynomialRLS

class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output, future_reward=None):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        :param future_reward: Future for reward computation (optional)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.future_reward = future_reward

    def __getitem__(self, item):
        """
        Enhanced indexing for GenerationBatchFuture objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            GenerationBatchFuture: For all indexing types except single integers
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return GenerationBatchFuture(
                epoch=self.epoch,
                batch=self.batch.slice(item.start, item.stop, item.step),
                gen_batch_output=self.gen_batch_output.slice(item.start, item.stop, item.step),
                future_reward=None
                )

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, int | np.integer | list | np.ndarray | torch.Tensor):
            if isinstance(item, list):
                item = np.array(item)
            return GenerationBatchFuture(
                epoch=self.epoch,
                batch=self.batch.select_idxs(item),
                gen_batch_output=self.gen_batch_output.select_idxs(item),
                future_reward=None
                )

        # # Case 3: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    # TODO: support when gen_batch_output is DataProtoFuture
    # TODO: support when future_reward is not None
    def concat(other: list["GenerationBatchFuture"]) -> "GenerationBatchFuture":
        """Aggregates multiple generation futures into a single large batch."""
        return GenerationBatchFuture(
            epoch=other[0].epoch,
            batch=DataProto.concat([obj.batch for obj in other]),
            gen_batch_output=DataProto.concat([obj.gen_batch_output for obj in other]),
            future_reward=None
        )

    def to(self, device) -> "GenerationBatchFuture":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        if self.gen_batch_output is not None:
            self.gen_batch_output = self.gen_batch_output.to(device)
        return self
                
    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (epoch, batch, gen_batch_result, future_reward)
                - epoch: Current epoch
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
                - future_reward: Future for reward computation if available, else None
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
        else:
            gen_batch_result = self.gen_batch_output

        return self.epoch, self.batch, gen_batch_result, self.future_reward


class Memory:
    """
    Replay Buffer for Off-policy RL. 
    Stores past generation results and provides reward prediction using RLS.
    """
    def __init__(self, k):
        self.k = k
        self.buffer = deque(maxlen=k)
        self.data_utilization = set()
        self.future_batch = None
        self.new_meta_info = None
        self.r = []
        self.ref_r = []
        self.ref_2_r = []
        self.beta = 1
        self.predictor = OnlinePolynomialRLS(degree=3, l2=1e-4)

    def update(self, item: 'GenerationBatchFuture'):
        '''
        Update stored future_batch to delay .get()
        '''
        self.future_batch = item
        
    def add(self, item: 'GenerationBatchFuture'):
        """
        Adds a GenerationBatchFuture item to the buffer.
        """
        if len(self.buffer) == self.k:
            old = self.buffer.pop()
            del old
            gc.collect()
        self.buffer.appendleft(item.to("cpu"))

    def get(self):
        """
        Get from self.future_batch
        """
        if self.future_batch:
            epoch, batch, gen_batch_output, future_reward = self.future_batch.get()
        return GenerationBatchFuture(epoch, batch, gen_batch_output, future_reward)

    def record_reward(self, step, reward):
        """Updates the reward predictor with the latest training feedback."""
        if isinstance(reward, DataProto):
            reward = reward["reward"]
        if isinstance(reward, torch.Tensor):
            reward = reward.sum(-1).mean().detach().item()
        self.r.append(reward)
        self.ref_r.append(reward * math.exp(reward))
        self.ref_2_r.append(math.exp(reward) / self.beta)
        print(f"actual: {reward}")
        if reward > 0:
            self.predictor.update(step, reward)
        
    def dummy_predict(self):
        return sum(self.r) / len(self.r)

    def dummy_predict_ref_r(self):
        return sum(self.ref_r) / len(self.ref_r)

    def dummy_predict_ref_2_r(self):
        return sum(self.ref_2_r) / len(self.ref_2_r)
        
    def predict(self, t):
        return self.predictor.predict(t)

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return f"UnlimitedBuffer with {len(self)} items."

