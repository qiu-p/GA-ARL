'''
time 
file & dir
'''

# import numpy as np
import os
import shutil
import random
import time
from os import path as osp

DEFAULT_QIUPUTILS_ENV_VAR_DICT = {
    "QIUPUTILS_BASE_DIR": "./",
    "QIUPUTILS_DEFAULT_OUTPUT_DIR": "outputs",
    "QIUPUTILS_DEFAULT_LOG_DIR": "logs",
    "QIUPUTILS_START_TIME": "time_not_set",
}

def get_time_str(tmpl='%Y%m%d_%H%M%S') -> str:
    '''
    %Y%m%d_%H%M%S
    '''
    return time.strftime(tmpl, time.localtime())

def set_qiuputils_env_var(env_var_dict: dict=None, **kwargs):
    '''
    set env var
    
    - if you use Ray, you can set env var in the main process and it will be inherited by all workers.
      - **call this function before initializing Ray**
      - to ensure that the environment variables are properly set for all workers
    '''
    if env_var_dict is None:
        env_var_dict = DEFAULT_QIUPUTILS_ENV_VAR_DICT

    # Update with any additional keyword arguments
    env_var_dict.update(kwargs)
    env_var_dict["QIUPUTILS_START_TIME"] = get_time_str()
    env_var_dict["QIUPUTILS_DEFAULT_OUTPUT_DIR"] = os.path.join(env_var_dict["QIUPUTILS_BASE_DIR"], env_var_dict["QIUPUTILS_DEFAULT_OUTPUT_DIR"])
    env_var_dict["QIUPUTILS_DEFAULT_LOG_DIR"] = os.path.join(env_var_dict["QIUPUTILS_BASE_DIR"], env_var_dict["QIUPUTILS_DEFAULT_LOG_DIR"])

    for key, value in env_var_dict.items():
        os.environ[key] = value

############################# 调整格式

def split_path(path: str) -> tuple[str, str, str]:
    '''
    - eg
      - 'AAA/BBB/CCC.txt'   -> 'AAA/BBB/', 'CCC', '.txt'
      - 'AAA/BBB/CCC'       -> 'AAA/BBB/', 'CCC', ''
      - 'AAA/BBB/'          -> 'AAA/BBB/', '',    ''
    '''
    pre_path = os.path.dirname(path)
    name_without_ext, ext = os.path.splitext(os.path.basename(path))

    return pre_path+'/', name_without_ext, ext

def trans_n_to_index(n: int, matrix_size: tuple | list) -> list:
    '''
    - Description:
      - 将 n 转换为其在矩阵中的坐标
    - Args:
      - n: 需要转换的整数索引
      - matrix_size: tuple 表示矩阵的每个维度的大小
    - Return:
      - result: 一个列表，表示 n 在矩阵中的多维索引
    - eg
      - 4, (2, 3) -> [1, 0]
    '''
    n = int(n)
    i = -1
    dimensionality = len(matrix_size)
    result = []
    while n != 0:
        result.insert(0, n % matrix_size[i])
        n = n // matrix_size[i]
        i -= 1
        dimensionality -= 1
    
    for _ in range(dimensionality):
        result.insert(0, 0)
    
    return result

def trans_pound_str(s_to_trans: str) -> str:
    '''
    - Description:
      - 将 s_to_trans 中的 #x 部分转换为具体要求的内推
      - If you want to express time, you can use '#t'.
      - If you just want to express '#", you can use '##'.
    '''
    detect_pound = False
    i = 0
    for char in s_to_trans:
        if detect_pound == True:
            str_to_replace = ''
            if char == 't':
                str_to_replace = get_time_str()
            elif char == '#':
                str_to_replace = '#'
            detect_pound = False
            if len(s_to_trans) == i+1:
                s_to_trans = s_to_trans[:i-1] + str_to_replace
            else:
                s_to_trans = s_to_trans[:i-1] + str_to_replace + s_to_trans[i+1:]
            i = i + len(str_to_replace) - 2
        elif char == '#':
            detect_pound = True
        i = i + 1
    return s_to_trans

#############################

def rename_dir_or_file(path, template='{}_archived_#t'):
    '''
    - Description:
      - Rename the name of the path.
    - Args:
      - path (str): dir path
      - template (str): the template to rename the path
      - delete (bool): if path exists, whether to delete it
    - Return:
      - Option(str): name after rename
    - Details:
      - If you want to express time, you can use '#t'.
      - If you just want to express '#", you can use '##'.
    '''
    if path[-1] == '/':
        path = path[:-1]
    if osp.exists(path):
        template = trans_pound_str(template)
        pre_path, name_without_ext, ext = split_path(path)
        new_name = pre_path + template.format(name_without_ext) + ext
        os.rename(path, new_name)
        return new_name
    else:
        return None

def path_add_suffix(path, suffix):
    '''
    - Description:
      - add suffix to dir name or file name
    - Args:
      - path (str): eg. AAA/BBB/CC.txt
      - index (str): eg. 1
    - Return:
        eg. AAA/BBB/CC_1.txt
    '''
    filename_tmpl = '{}_' + suffix
    rename_dir_or_file(path, filename_tmpl)

def make_dir(dir_path, mode='e', template='{}_archived_#t'):
    """
    - Description:
      - mkdirs
    - Args:
      - path (str): Folder path.
      - mode (str): 'e' 'i' 'r' or 'd'.
        - e: (default) throw ERROR if dir has existed
        - i: (ignore) if dir has existed, just return and do nothing
        - r: (rename) rename the old dir if dir has existed
        - d: (delete) delete the old dir if dir has existed
        - m: (move) rename the old dir and move it to a sub dir "legacy" if dir has existed
      - template (str): the template to rename the path
        - If you want to express time, you can use '#t'.
        - If you just want to express '#", you can use '##'. 
        - default: '{}_archived_#t'
    - Return:
      - is_exist (bool): whether the dir has existed
    """
    is_exist = False
    if osp.exists(dir_path):
        is_exist = True
        if mode == 'r':
            rename_dir_or_file(dir_path, template)
        elif mode == 'd':
            shutil.rmtree(dir_path)
        elif mode == 'e':
            raise FileExistsError(f'the dir {dir_path} has existed')
        elif mode == 'i':
            return is_exist
        elif mode == 'm':
            new_name = rename_dir_or_file(dir_path, template)

            parent_dir = osp.dirname(new_name)
            legacy_dir = osp.join(parent_dir, 'legacy')
            os.makedirs(legacy_dir, exist_ok=True)
            
            # 计算移动后的目标完整路径
            target_path = osp.join(legacy_dir, osp.basename(new_name))
            
            # 执行移动操作
            if osp.exists(target_path):
                raise FileExistsError(f"the dir {target_path} has existed")
            else:
                shutil.move(new_name, target_path)
        else:
            raise ValueError('\'{}\' is not a mode. Mode selection: e, r, d.'.format(mode))
    # exist_ok：是否在目录存在时触发异常。
    # exist_ok = False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；
    # exist_ok = True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
    os.makedirs(dir_path)
    return is_exist

def scan_dir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scan_dir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scan_dir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scan_dir(dir_path, suffix=suffix, recursive=recursive)

##############################################

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file size.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'

def get_from_dict(dict, key, replace_none = None):
    """ 
    get value from dictionary
    Aegs:
        dict (dict):
        key (int, str, ...):
        replace_none: the value to replace the return if the original value is None
    Return:
        (bool): whethe the key exists
        (value): value 
    """
    if key not in dict:
        return False, None
    else:
        value = dict.get(key)
        if value == None:
            value = replace_none
        return True, value
