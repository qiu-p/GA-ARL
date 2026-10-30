import logging
import datasets
import torch
import numpy as np
from torch.utils.data import Dataset

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

def default_get_field_for_source(data_source: str) -> str:
    source_to_field = {
        'codegen__leetcode2k': 'code',
        'codegen__taco': 'code',
        'codegen__primeintellect': 'code',
        'codegen__livecodebench': 'code',
        'math__merged_deduped_dapo_or1_dataset': 'math',
        'math__deepscaler_preview': 'math',
        'stem_web': 'science',
        'logic__ordering_puzzle_dataset': 'logic',
        'logic__zebra_puzzle_dataset': 'logic',
        'logic__graph_logical_dataset': 'logic',
        # for eval
        "math__aime2024": 'math',
        "math__gsm8k": 'math',
        "math__amc_repeated_4x": 'math',
        "math__math": 'math',
        "stem__supergpqa": 'science',
        "math__aime_repeated_8x": 'math',
        "codegen__humaneval": 'code',
        'codegen__mbpp': 'code',
    }
    
    field = "unknown"
    if data_source in source_to_field:
        field = source_to_field[data_source]
    elif data_source.startswith("math_"):
        field = "math"
    elif data_source.startswith("codegen_"):
        field = "code"
    elif data_source.startswith("logic_"):
        field = "logic"

    return field

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class ConcateMultiDataset(Dataset):
    '''
    A dataset that concatenates multiple datasets and allows indexing across them.
    '''
    def __init__(self, datasets_list):
        self.datasets_list = datasets_list
        # 累积每个子dataset的长度，方便索引映射
        self.cum_lengths = []
        total = 0
        for d in datasets_list:
            total += len(d)
            self.cum_lengths.append(total)
        self.total_len = total

    def select(self, indices):
        return SubsetDataset(self, indices)
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # 找到 idx 属于哪个子dataset
        for ds_idx, cum_len in enumerate(self.cum_lengths):
            if idx < cum_len:
                if ds_idx == 0:
                    inner_idx = idx
                else:
                    inner_idx = idx - self.cum_lengths[ds_idx - 1]
                return self.datasets_list[ds_idx][inner_idx]
        raise IndexError("Index out of range")

class RLHFMutiDataset(RLHFDataset):
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            print(f"Loading dataset from {parquet_file}...")
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframe = self.maybe_filter_out_long_prompts(dataframe)
            dataframes.append(dataframe)
        self.dataframe: ConcateMultiDataset = ConcateMultiDataset(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        eg.
        dataframe = [data0, data1, data2, ...]
        data = {
            "data_source": "gsm8k",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": _},
            "extra_info": {"split": "train", "index": _, "level": _}
        }
        return row_dict = {
            "input_ids": tensor([...]),
            "attention_mask": tensor([0, 0, 0, ..., 1, 1, 1]),
            "position_ids": tensor([0, 0, 0, ..., 100, 101, 102]),

            "data_source": "gsm8k",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": _},
            "extra_info": {"split": "train", "index": _, "level": _},

            "raw_prompt_ids": List([...]),
            "index": _,
            "tools_kwargs": {},
            "interaction_kwargs": {},

            "field": "_",
            "dataframe_index": _
        }
        """
        row_dict: dict = self.dataframe[item]
        # MY: only keep 5 keys
        keys_to_keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
        row_dict = {k: row_dict[k] for k in keys_to_keep if k in row_dict}
        
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        # MY: add field info
        # data_source = row_dict['data_source']
        # field = row_dict.get("extra_info", {}).get("field", None)
        # if field == None:
        #     field = self.get_field_for_source_func(data_source)
        # row_dict["field"] = field
        row_dict["dataframe_index"] = item
        
        return row_dict
