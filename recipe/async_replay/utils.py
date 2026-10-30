# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


import logging
import os
import random

import numpy as np
import torch
from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import AdvantageEstimator

logger = logging.getLogger(__name__)


def set_deterministic(seed=42, set_base=True, set_numpy=True, set_accelerator=True) -> None:
    """Set deterministic seeds for CUDA-oriented recipe execution."""
    if set_base:
        random.seed(a=seed)

    if set_numpy:
        np.random.seed(seed=seed)

    torch.manual_seed(seed=seed)

    if set_accelerator and torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)

    try:
        torch.use_deterministic_algorithms(mode=True)
    except Exception as exc:
        logger.warning("Failed to enable deterministic algorithms: %s", exc)


def log_accelerator_infos(log):
    """Log CUDA accelerator and distributed state."""
    if not torch.cuda.is_available():
        log.info("No CUDA accelerator is available.")
        return

    device_info = (
        f"torch.cuda: current_device: {torch.cuda.current_device()} "
        f"device_count: {torch.cuda.device_count()} "
        f"get_device_name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        distributed_info = (
            f"torch.distributed: get_rank: {torch.distributed.get_rank()} "
            f"get_world_size: {torch.distributed.get_world_size()}"
        )
    else:
        distributed_info = None

    rank_info = (
        f"os.environ: RANK: {os.getenv('RANK', None)} "
        f"LOCAL_RANK: {os.getenv('LOCAL_RANK', None)} "
        f"WORLD_SIZE: {os.getenv('WORLD_SIZE', None)} "
        f"VISIBLE_DEVICES: {visible_devices}"
    )
    msg = f"CUDA is available.\n{device_info}\n{rank_info}"
    if distributed_info:
        msg += f"\n{distributed_info}"
    log.info(msg)


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic"""
    if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    elif config.algorithm.adv_estimator in [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.REINFORCE_PLUS_PLUS,
        # AdvantageEstimator.REMAX, # TODO:REMAX advantage estimator is not yet supported in this recipe
        AdvantageEstimator.RLOO,
        AdvantageEstimator.OPO,
        AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        AdvantageEstimator.GPG,
    ]:
        return False
    else:
        raise NotImplementedError


def poly_predict_least_squares(ts, rs, degree, reg=0.0):
    """
    ts: list/array of t values (integers but can be any distinct x)
    rs: list/array of r values
    degree: polynomial degree (d)
    reg: ridge regularization lambda (default 0: none)
    returns predicted value at x = max(ts)+1
    """
    x = np.asarray(ts)
    y = np.asarray(rs)
    # Build Vandermonde (columns: 0..degree)
    V = np.vander(x, N=degree+1, increasing=True)  # shape (n, d+1)
    # Solve regularized normal equations: (V^T V + λI) w = V^T y
    A = V.T @ V
    b = V.T @ y
    if reg > 0:
        A = A + reg * np.eye(A.shape[0])
    coeffs = np.linalg.solve(A, b)  # coeffs[0] + coeffs[1]*x + ...
    xpred = np.array([max(x)+1])
    Vpred = np.vander(xpred, N=degree+1, increasing=True)
    return float(Vpred @ coeffs)

class OnlinePolynomialRLS:
    """
    在线多项式拟合 (Recursive Least Squares version)
    用于动态预测 r(t)，可随时间更新。
    """

    def __init__(self, degree, l2=0.0):
        self.d = degree + 1          # 特征维数 = 阶数+1
        self.lam = l2
        # 初始化同前面多元 RLS
        self.P = np.eye(self.d) / self.lam if self.lam > 0 else np.eye(self.d) * 1e8
        self.w = np.zeros(self.d)

    # ---------- 私有：1-D -> 多项式特征 ----------
    def _poly_feat(self, x):
        """标量 x -> [1, x, x², …, x^degree]"""
        return np.array([x ** k for k in range(self.d)], dtype=float)

    # ---------- 在线更新 ----------
    def update(self, x, y):
        """
        来一条新样本 (x, y)
        x: float
        y: float
        """
        phi = self._poly_feat(x)     # 1×d 向量
        # RLS 更新
        Px = self.P @ phi
        den = 1.0 + phi @ Px
        K = Px / den
        e = y - phi @ self.w
        self.w += K * e
        self.P -= np.outer(K, Px)

    # ---------- 预测 ----------
    def predict(self, x):
        return self._poly_feat(x) @ self.w

    # ---------- 方便打印 ----------
    def __str__(self):
        return "w=" + np.array2string(self.w, precision=4)
