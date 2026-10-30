'''
usage:

opt = parse_yaml()

or

args = parse_options()
opt = parse_yaml(args.opt, args.force_yaml)

or

parser = argparse.ArgumentParser()
parser.add_argument('--your_args', type=str, help='your args')
args = parse_options(parser)
opt = parse_yaml(args.opt, args.force_yaml)
'''
import argparse
import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    # replace() 方法把字符串中的 old（旧字符串） 替换成 new(新字符串)，如果指定第三个参数max，则替换不超过 max 次
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # str
    return value


def parse_options(parser=None):
    if parser==None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file. Splited by \',\'. Examples: foder/path1;path2')
    parser.add_argument(
        '--force_yaml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path=None, force_yaml=None):
    '''
    usage:

    opt = parse_yaml()

    or

    args = parse_options()
    opt = parse_yaml(args.opt, args.force_yaml)

    or

    parser = argparse.ArgumentParser()
    parser.add_argument('--your_args', type=str, help='your args')
    args = parse_options(parser)
    opt = parse_yaml(args.opt, args.force_yaml)
    '''
    if yaml_path is None:
        args = parse_options()
        yaml_path = args.opt
        force_yaml = args.force_yaml

    # parse yml to dict
    with open(yaml_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # force to update yml options
    if force_yaml is not None:
        for entry in force_yaml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    return opt

def parse_yamls(yaml_paths, force_yaml=None):
    ''' 
    return:
        opts: eg.  [(opt1, this_yaml_dir1), 
                    (opt2, this_yaml_dir2), 
                    ...]
    '''
    yaml_paths = yaml_paths.split(',')
    opts = [] # (opt, this_yaml_dir)
    for yaml_path in yaml_paths:
        # parse yml to dict
        with open(yaml_path, mode='r') as f:
            opt = yaml.load(f, Loader=ordered_yaml()[0])

        # force to update yml options
        if force_yaml is not None:
            for entry in force_yaml:
                # now do not support creating new keys
                keys, value = entry.split('=')
                # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
                keys, value = keys.strip(), value.strip()
                value = _postprocess_yml_value(value)
                eval_str = 'opt'
                for key in keys.split(':'):
                    eval_str += f'["{key}"]'
                eval_str += '=value'
                # using exec function
                exec(eval_str)

        opts.append((opt, osp.dirname(yaml_path)))

    return opts
''