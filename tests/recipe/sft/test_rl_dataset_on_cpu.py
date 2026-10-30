import pandas as pd
from omegaconf import OmegaConf

from recipe.sft.rl_dataset import (
    SFTRolloutEvalDataset,
    SFTTrainDataset,
    create_sft_dataset,
    create_sft_sampler,
    normalize_data_paths,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        assert tokenize is False
        if isinstance(messages, str):
            text = messages
        else:
            text = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        return text + ("\nassistant:" if add_generation_prompt else "")

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        import torch

        ids = [max(1, (ord(char) % 97) + 1) for char in text]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        return [max(1, (ord(char) % 97) + 1) for char in text]


def write_parquet(path):
    dataframe = pd.DataFrame(
        [
            {
                "data_source": "math__gsm8k",
                "prompt": [{"role": "user", "content": "What is 1+1?"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "2"},
                "extra_info": {"question": "What is 1+1?", "answer": "2", "index": 0},
            },
            {
                "data_source": "logic__zebra_puzzle_dataset",
                "prompt": [{"role": "user", "content": "A before B?"}],
                "ability": "logic",
                "reward_model": {"style": "rule", "ground_truth": "A"},
                "extra_info": {"question": "A before B?", "answer": "A", "index": 1},
            },
        ]
    )
    dataframe.to_parquet(path)


def test_normalize_data_paths_expands_directory(tmp_path):
    file_path = tmp_path / "part.parquet"
    write_parquet(file_path)

    paths = normalize_data_paths(str(tmp_path))

    assert paths == [str(file_path)]


def test_train_dataset_builds_sft_tensors_from_nested_fields(tmp_path):
    file_path = tmp_path / "train.parquet"
    write_parquet(file_path)
    config = OmegaConf.create(
        {
            "prompt_key": "extra_info",
            "response_key": "extra_info",
            "prompt_dict_keys": ["question"],
            "response_dict_keys": ["answer"],
            "max_length": 128,
            "truncation": "error",
            "use_shm": False,
        }
    )

    dataset = SFTTrainDataset([str(file_path)], tokenizer=DummyTokenizer(), config=config)
    item = dataset[0]

    assert {"input_ids", "attention_mask", "position_ids", "loss_mask"}.issubset(item)
    assert item["input_ids"].shape[0] == 128
    assert item["loss_mask"].sum().item() > 0
    assert item["data_source"] == "math__gsm8k"
    assert item["reward_model"]["ground_truth"] == "2"


def test_eval_dataset_returns_prompt_only_rollout_fields(tmp_path):
    file_path = tmp_path / "val.parquet"
    write_parquet(file_path)
    config = OmegaConf.create(
        {
            "eval_prompt_key": "prompt",
            "max_prompt_length": 96,
            "truncation": "error",
            "filter_overlong_prompts": False,
            "use_shm": False,
        }
    )

    dataset = SFTRolloutEvalDataset([str(file_path)], tokenizer=DummyTokenizer(), config=config)
    item = dataset[0]

    assert {"input_ids", "attention_mask", "position_ids", "raw_prompt_ids"}.issubset(item)
    assert item["input_ids"].shape[0] == 96
    assert item["data_source"] == "math__gsm8k"
    assert item["field"] == "math"
    assert item["dataframe_index"] == 0


def test_create_sft_dataset_selects_train_or_eval_dataset(tmp_path):
    file_path = tmp_path / "data.parquet"
    write_parquet(file_path)
    config = OmegaConf.create(
        {
            "prompt_key": "extra_info",
            "response_key": "extra_info",
            "prompt_dict_keys": ["question"],
            "response_dict_keys": ["answer"],
            "eval_prompt_key": "prompt",
            "max_length": 128,
            "max_prompt_length": 96,
            "truncation": "error",
            "filter_overlong_prompts": False,
            "shuffle": False,
            "sampler": None,
            "custom_cls": {"path": None, "name": None},
            "use_shm": False,
        }
    )

    train_dataset = create_sft_dataset([str(file_path)], config, DummyTokenizer(), is_train=True)
    val_dataset = create_sft_dataset([str(file_path)], config, DummyTokenizer(), is_train=False)

    assert isinstance(train_dataset, SFTTrainDataset)
    assert isinstance(val_dataset, SFTRolloutEvalDataset)
    assert create_sft_sampler(config, train_dataset).__class__.__name__ == "SequentialSampler"
