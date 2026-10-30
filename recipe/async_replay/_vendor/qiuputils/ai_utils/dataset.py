import os
import glob

def normalize_dataset_paths(dataset_paths, extension=".parquet"):
    """
    检查并标准化 dataset_paths：
    - 如果 dataset_paths 是 list，则直接返回；
    - 如果是单个路径：
        * 若为文件：返回 [dataset_paths]
        * 若为文件夹：返回该文件夹下所有 {extension} 文件路径列表
    """
    # 如果本来就是 list
    if isinstance(dataset_paths, list):
        return dataset_paths

    # 否则要求是 str 类型路径
    if not isinstance(dataset_paths, str):
        raise ValueError(f"Invalid dataset_paths type: {type(dataset_paths)}, expected str or list")

    # 若路径不存在
    if not os.path.exists(dataset_paths):
        raise FileNotFoundError(f"Path not found: {dataset_paths}")

    # 若是文件，则保持不变
    if os.path.isfile(dataset_paths):
        return [dataset_paths]

    # 若是文件夹，则读取所有 {extension} 文件
    if os.path.isdir(dataset_paths):
        extension_files = sorted(glob.glob(os.path.join(dataset_paths, f"*{extension}")))
        if not extension_files:
            raise FileNotFoundError(f"No {extension} files found in folder: {dataset_paths}")
        return extension_files

    # 其他类型（如管道、符号链接等）直接报错
    raise ValueError(f"Unsupported path type: {dataset_paths}")