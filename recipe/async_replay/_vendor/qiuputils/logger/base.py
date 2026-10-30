import logging
import time
import sys
import os
from typing import Optional

initialized_logger = {}
all_loggers = {}

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        # 计算从程序启动到日志记录的秒数
        elapsed_seconds = round(record.created - self.start_time)
        # 构造 prefix
        prefix = f"{record.levelname} - {time.strftime('%x %X', time.localtime(record.created))} - {elapsed_seconds}"
        # 获取原始消息
        message = record.getMessage()
        
        if not message:
            return prefix + " - "

        # 计算需要填充的空格数（prefix + " - " 的长度）
        indent = " " * (len(prefix) + 3)
        # 将消息中的每个换行符替换为 "\n" + 缩进
        formatted_message = message.replace("\n", "\n" + indent)

        return f"{prefix} - {formatted_message}"



class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # flush() is generally used to force the system to write any buffered data to the output
        pass
        
        # not sure below
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        # self.level(sys.stderr)

class LoggerWrapper:
    '''
    LoggerWrapper 的 Docstring
    
    添加了一些方法
    保留方法
        debug
        info
        warning
        error
    '''
    def __init__(
            self,
            logger: logging.Logger,
            separator_num=10
        ):
        self.logger = logger
        self.separator_num = separator_num
        self.recorded_time = {}

    def debug(self, msg, indentation_num=0):
        indentation = "\t" * indentation_num
        self.logger.debug(indentation + msg)

    def info(self, msg, indentation_num=0):
        indentation = "\t" * indentation_num
        self.logger.info(indentation + msg)

    def warning(self, msg, indentation_num=0):
        indentation = "\t" * indentation_num
        self.logger.warning(indentation + msg)

    def error(self, msg, indentation_num=0):
        indentation = "\t" * indentation_num
        self.logger.error(indentation + msg)

    # ========= 新增的业务方法 =========

    def record_time(self, time_tag="base"):
        self.recorded_time[time_tag] = time.time()

    def log_separator(self, msg, indentation_num=0):
        indentation = "\t" * indentation_num
        self.logger.info(indentation + "="*self.separator_num + f" {msg} " + "="*self.separator_num)

    def log_separator_end(self, msg, indentation_num=0):
        indentation = "\t" * indentation_num
        self.logger.info(indentation + "="*self.separator_num + "="*len(f" {msg} ") + "="*self.separator_num)

    def log_start(self):
        self.log_separator("START")

    def log_end(self):
        self.log_separator("END")

    def log_recorded_time(self, time_tag="base", indentation_num=0):
        indentation = "\t" * indentation_num
        self.logger.info(f"{indentation}{time_tag} time: {self.recorded_time[time_tag]:.2f}s")

    def log_cur_time(self, indentation_num=0):
        indentation = "\t" * indentation_num
        self.logger.info(f"{indentation}current time: {time.time():.2f}s")

    def log_elapsed(self, start_time_tag="base", indentation_num=0):
        indentation = "\t" * indentation_num
        elapsed = time.time() - self.recorded_time[start_time_tag]
        self.logger.info(f"{indentation}{start_time_tag} elapsed: {elapsed:.2f}s")

    def log_batch_start(self, step, total=None, indentation_num=0):
        indentation = "\t" * indentation_num
        if total:
            self.logger.info(f"{indentation}[Batch {int(step)}/{int(total)}] Start")
        else:
            self.logger.info(f"{indentation}[Batch {int(step)}] Start")
    
    def log_cuda_infos(self, indentation_num=0):
        import torch
        indentation = "\t" * indentation_num
        if torch.cuda.is_available():
            cuda_info = f"torch.cuda: current_device: {torch.cuda.current_device()}  device_count: {torch.cuda.device_count()}  get_device_name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
            if torch.distributed.is_initialized():
                distributed_info = f"torch.distributed: get_rank: {torch.distributed.get_rank()}  get_world_size: {torch.distributed.get_world_size()}"
            else:
                distributed_info = None
            rank_info = f"os.environ: RANK: {os.getenv('RANK', None)}  LOCAL_RANK: {os.getenv('LOCAL_RANK', None)}  WORLD_SIZE: {os.getenv('WORLD_SIZE', None)}  CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', None)}"
            finel_str = f"{indentation}CUDA is available."
            if cuda_info:
                finel_str += f"\n{cuda_info}"
            if distributed_info:
                finel_str += f"\n{distributed_info}"
            if rank_info:
                finel_str += f"\n{rank_info}"
            self.logger.info(finel_str)
        else:
            self.logger.info(f"{indentation}CUDA is not available.")
    
    def log_dict(self, dict_data: dict, title: Optional[str]=None, indentation_num=0):
        indentation = "\t" * indentation_num
        log_str = ""
        if title:
            log_str += f"{indentation}{title}:\n"
        for key, value in dict_data.items():
            log_str += f"{indentation}{key}: {value}\n"
        log_str = log_str.rstrip("\n")  # 去除最后一个换行符
        self.logger.info(log_str)

def _init_logger(logger: logging.Logger, log_level=logging.DEBUG, file_path=None, is_console=False, change_stdout_stderr=False):
    """
    Create a logger.
    """
    if file_path==None and not is_console:
        raise Exception('Logger has no handler')
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if file_path is not None:
        file_folder = os.path.dirname(file_path)
        if file_folder == '':
            file_folder = './'
        if not os.path.exists(file_folder):
            # exist_ok：是否在目录存在时触发异常。
            # exist_ok = False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；
            # exist_ok = True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
            os.makedirs(file_folder)
        file_handler = logging.FileHandler(file_path, "a", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    if is_console:
        # create console handler and set level to info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)
 
    # create logger and set level to debug
    logger.handlers = []
    logger.setLevel(log_level)
    logger.propagate = False
    if file_path is not None:
        logger.addHandler(file_handler)
    if is_console:
        logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    if change_stdout_stderr:
        sys.stdout = LoggerWriter(logger.debug)
        sys.stderr = LoggerWriter(logger.warning)
    
    return LoggerWrapper(logger)

def get_all_loggers():
    return all_loggers

def get_logger(logger_name='base', log_level=logging.DEBUG, log_file=None, is_console=False, change_stdout_stderr=False) -> LoggerWrapper:
    '''
    if the logger has exsited, then just return
    if the logger doesn't exsit, then create the logger and return it
    change_stdout_stderr: change sys.stdout and sys.stderr to logger output
    '''
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return all_loggers[logger_name]
    else:
        logger = logging.getLogger(logger_name)
        if log_file is None:
            start_time_str = os.environ.get('QIUPUTILS_START_TIME', None)
            if start_time_str is None:
                file_name = f"{logger_name}_{os.getpid()}.log"
            else:
                file_name = f"{logger_name}_{os.environ.get('QIUPUTILS_START_TIME', None)}_{os.getpid()}.log"
            log_file = os.path.join(
                os.environ.get("QIUPUTILS_DEFAULT_LOG_DIR", None),
                file_name
            )
        logger = _init_logger(logger, log_level, log_file, is_console, change_stdout_stderr)
        initialized_logger[logger_name] = True
        all_loggers[logger_name] = logger
        return logger
