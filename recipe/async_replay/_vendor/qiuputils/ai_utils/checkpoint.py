import os
import json
import math
import shutil

class CheckpointManager:
    """
    负责：
    1. 记录每个 metric 的历史最优 checkpoint；
    2. 判断当前 val_metrics 是否刷新 best；
    3. 清理旧 checkpoint；
    4. 清理时优先保留 best checkpoint。
    """

    def __init__(
        self,
        base_dir: str,
        meta_filename: str = "best_checkpoint_metrics.json",
        logger=None,
        indentation_num=2,
        metrics_startswith="mean",
        save_priority: list[str] | None = None,
    ):
        self.base_dir = base_dir
        self.meta_filename = meta_filename
        self.logger = logger
        self.indentation_num = indentation_num
        self.metrics_startswith = metrics_startswith
        self.save_priority = save_priority

    def _log(self, msg: str):
        print(msg)
        if self.logger is not None:
            try:
                self.logger.info(msg, indentation_num=self.indentation_num)
            except TypeError:
                self.logger.info(msg)

    def _to_float(self, x):
        try:
            if hasattr(x, "item"):
                x = x.item()
            x = float(x)
        except Exception:
            return None

        if not math.isfinite(x):
            return None

        return x

    @property
    def meta_path(self):
        return os.path.join(self.base_dir, self.meta_filename)
    
    def set_metrics_startswith(self, metrics_startswith):
        self.metrics_startswith = metrics_startswith

    def set_save_priority(self, save_priority):
        self.save_priority = save_priority

    def load_meta(self) -> dict:
        if not os.path.exists(self.meta_path):
            return {
                "best_metrics": {},
                "ckpt_metrics": {},
            }

        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        meta.setdefault("best_metrics", {})
        meta.setdefault("ckpt_metrics", {})

        return meta

    def save_meta(self, meta: dict):
        os.makedirs(self.base_dir, exist_ok=True)

        tmp_path = self.meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        os.replace(tmp_path, self.meta_path)

    def extract_metrics(self, val_metrics: dict, startswith="mean") -> dict[str, float]:
        """
        从 val_metrics 中提取所有 mean 项。

        val_metrics 的 key 形式大致是：
        val-core/{data_source}/{var_name}/{metric_name}
        val-aux/{data_source}/{var_name}/{metric_name}

        例如：
        val-core/math/acc/mean@8
        val-core/math/reward/mean@16
        val-aux/code/pass_rate/mean@4
        """
        metrics = {}

        if not val_metrics:
            return metrics

        for key, value in val_metrics.items():
            parts = key.split("/", 3) # 按照分隔符 / 进行分割，并且最多只分割 3 次
            if len(parts) < 4:
                continue

            metric_name = parts[3]

            if not metric_name.startswith(startswith):
                continue

            value = self._to_float(value)
            if value is None:
                continue

            metrics[key] = value

        return metrics

    def get_improved_metrics(self, val_metrics: dict) -> dict[str, dict]:
        """
        判断当前 val_metrics 中哪些 metric 刷新了历史最优。

        注意：
        这里默认所有 metric 都是越大越好。
        例如 acc 、reward 一般都是越大越好。
        """
        metrics = self.extract_metrics(val_metrics, startswith=self.metrics_startswith)
        if not metrics:
            return {}

        meta = self.load_meta()
        best_metrics = meta.get("best_metrics", {})

        improved = {}

        for metric_key, current_value in metrics.items():
            old_item = best_metrics.get(metric_key)
            old_best = None if old_item is None else old_item.get("best")

            if old_best is None or current_value > old_best:
                improved[metric_key] = {
                    "old_best": old_best,
                    "new_best": current_value,
                }

        return improved

    def update_after_save(
        self,
        step: int,
        ckpt_path: str,
        val_metrics: dict | None,
    ):
        """
        checkpoint 保存成功后调用。

        记录两类信息：
        1. ckpt_metrics：当前 step 对应的所有 metric；
        2. best_metrics：每个 metric 当前最优的 step / path / value。
        """
        metrics = self.extract_metrics(val_metrics, startswith=self.metrics_startswith)
        if not metrics:
            return

        meta = self.load_meta()
        best_metrics = meta.setdefault("best_metrics", {})
        ckpt_metrics = meta.setdefault("ckpt_metrics", {})

        step = int(step)
        step_key = str(step)

        ckpt_metrics[step_key] = {
            metric_key: value
            for metric_key, value in metrics.items()
        }

        for metric_key, current_value in metrics.items():
            old_item = best_metrics.get(metric_key)
            old_best = None if old_item is None else old_item.get("best")

            if old_best is None or current_value > old_best:
                best_metrics[metric_key] = {
                    "best": current_value,
                    "step": step,
                    "path": ckpt_path,
                }

        self.save_meta(meta)

    def get_protected_best_steps(self, meta: dict | None = None) -> set[int]:
        """
        获取所有需要保护的 best checkpoint step。

        不同 metric 分别维护最优值，所以可能有多个 best step。
        """
        if meta is None:
            meta = self.load_meta()

        protected_steps = set()

        for item in meta.get("best_metrics", {}).values():
            step = item.get("step", None)
            if step is None:
                continue

            try:
                protected_steps.add(int(step))
            except Exception:
                continue

        return protected_steps

    def checkpoint_score(self, step: int, meta: dict | None = None) -> float:
        """
        计算某个 checkpoint 的代表分数，用于清理排序。

        当前策略：
        - 如果该 checkpoint 有多个 metric，取 max 作为代表分数；
        - 如果没有记录，则认为分数最低，优先删除。

        如果你想按某个固定指标排序，比如 acc ，可以在这里改。
        """
        if meta is None:
            meta = self.load_meta()

        step_metrics = meta.get("ckpt_metrics", {}).get(str(step), {})
        values = []

        for value in step_metrics.values():
            value = self._to_float(value)
            if value is not None:
                values.append(value)

        if not values:
            return float("-inf")

        return max(values)

    def list_checkpoint_dirs(self) -> list[tuple[int, str]]:
        """
        找到 base_dir 下所有 global_step_xxx 目录。
        """
        if not os.path.exists(self.base_dir):
            return []

        ckpt_dirs = []

        for name in os.listdir(self.base_dir):
            if not name.startswith("global_step_"):
                continue

            path = os.path.join(self.base_dir, name)
            if not os.path.isdir(path):
                continue

            try:
                step = int(name.split("global_step_")[-1])
            except ValueError:
                continue

            ckpt_dirs.append((step, path))

        ckpt_dirs.sort(key=lambda x: x[0])
        return ckpt_dirs

    def cleanup_old_checkpoints(
        self,
        max_ckpt_to_keep: int | None,
    ):
        """
        清理旧 checkpoint。

        策略：
        1. best checkpoint 优先保留；
        2. 若 self.save_priority 为 None，best checkpoint 数量超过 max_ckpt_to_keep
           时，允许实际保留数量超过上限；
        3. 若 self.save_priority 为 list（val_metrics 完整 str 列表，按优先级从高到低排列，
           如 ["val-core/codegen livecodebench/acc/pass@8", ...]），
           则按优先级决定保留哪些 best checkpoint，总数严格不超过 max_ckpt_to_keep；
        4. 非 best checkpoint 中，优先删除分数低的；
        5. 分数一样时，优先删除更旧的 step。
        """
        if max_ckpt_to_keep is None or max_ckpt_to_keep <= 0:
            return

        if not os.path.exists(self.base_dir):
            return

        meta = self.load_meta()
        ckpt_dirs = self.list_checkpoint_dirs()

        if len(ckpt_dirs) <= max_ckpt_to_keep:
            return

        existing_steps = {step for step, _ in ckpt_dirs}

        if self.save_priority is None:
            # 原有逻辑：best checkpoint 优先保留，允许超过上限
            protected_steps = self.get_protected_best_steps(meta)
            protected_existing_steps = protected_steps & existing_steps
            real_keep_limit = max(max_ckpt_to_keep, len(protected_existing_steps))
        else:
            # 按优先级决定保留哪些 best checkpoint
            priority_map = {mk: i for i, mk in enumerate(self.save_priority)}
            default_priority = len(self.save_priority)

            # 构建 step -> 该 step 是哪些 metric 的 best
            step_to_best_metrics: dict[int, list[str]] = {}
            for metric_key, item in meta.get("best_metrics", {}).items():
                step = item.get("step")
                if step is None:
                    continue
                step = int(step)
                if step not in step_to_best_metrics:
                    step_to_best_metrics[step] = []
                step_to_best_metrics[step].append(metric_key)

            # 为每个 protected step 计算最高优先级（取所有 best metric 中的最小 index）
            protected_with_priority = []
            for step in existing_steps:
                best_metrics_for_step = step_to_best_metrics.get(step)
                if best_metrics_for_step is None:
                    continue
                best_priority = min(
                    priority_map.get(mk, default_priority)
                    for mk in best_metrics_for_step
                )
                protected_with_priority.append((step, best_priority))

            # 按优先级从高到低（index 小优先），同优先级按分数从高到低
            protected_with_priority.sort(
                key=lambda x: (x[1], -self.checkpoint_score(x[0], meta))
            )

            # 最多保留 max_ckpt_to_keep 个 best checkpoint
            protected_existing_steps = {
                step for step, _ in protected_with_priority[:max_ckpt_to_keep]
            }
            real_keep_limit = max_ckpt_to_keep

        excess = len(ckpt_dirs) - real_keep_limit
        if excess <= 0:
            return

        deletable_ckpts = [
            (step, path)
            for step, path in ckpt_dirs
            if step not in protected_existing_steps
        ]

        if not deletable_ckpts:
            return

        # 删除优先级：
        # 1. metric 分数越低越先删；
        # 2. step 越小越先删。
        deletable_ckpts.sort(
            key=lambda x: (
                x[0],
                self.checkpoint_score(x[0], meta),
            )
        )

        delete_num = min(excess, len(deletable_ckpts))

        for step, path in deletable_ckpts[:delete_num]:
            self._log(f"[Checkpoint] Removing old checkpoint: {path}")
            shutil.rmtree(path, ignore_errors=True)

            # 同步删除该 step 的 metric 记录
            meta.get("ckpt_metrics", {}).pop(str(step), None)

        self.save_meta(meta)