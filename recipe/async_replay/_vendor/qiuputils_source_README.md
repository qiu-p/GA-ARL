# README

- usage:
  - `pip install -e .`
  - `from qiuputils import <package>`

## logger

- 使用
  - `from qiuputils import get_logger`
  - 默认参数 `get_logger(logger_name='base', log_level=logging.DEBUG, log_file=None, is_console=False, change_stdout_stderr=False)`
    - logger_name 为该 logger 的名字
    - ...
  - 获取过的 logger 会存储在 all_loggers 中，下次 get_logger 传入该 logger 的 logger_name，会直接返回之前得到的logger
    - 注意不同进程由于内存不共享，所以 all_loggers 不会共享，进程之间的 logger 是不互通的
  - 若第一次使用获取某个 logger 时没有传入 log_file，则会设定默认位置
    - 读取 os.environ[QIUPUTILS_DEFAULT_LOG_DIR]
    - 默认路径为 os.path.join(os.environ.get("QIUPUTILS_DEFAULT_LOG_DIR", None), f"{logger_name}_{os.getpid()}.log")
    - 若 os.environ[QIUPUTILS_DEFAULT_LOG_DIR] 未设置或为 None，则会报错

### Usage en

- Import the logger utility:

  ```python
  from qiuputils import get_logger
  ```

- The `get_logger` function has the following default signature:

  ```python
  get_logger(
      logger_name='base',
      log_level=logging.DEBUG,
      log_file=None,
      is_console=False,
      change_stdout_stderr=False
  )
  ```

  - `logger_name` (str): The unique name identifying this logger instance.
  - `log_level` (int): The logging severity threshold (e.g., `logging.DEBUG`, `logging.INFO`). Only messages at or above this level will be processed.
  - `log_file` (str or None): Optional path to a file where logs should be written. If not provided, a default log file path may be used (see below).
  - `is_console` (bool): If `True`, log messages will also be printed to the console (stdout).
  - `change_stdout_stderr` (bool): If `True`, redirects `sys.stdout` and `sys.stderr` to the logger for capturing print statements and errors.

- **Logger Caching**:  
  All created loggers are stored in an internal dictionary `all_loggers`. If `get_logger` is called again with the same `logger_name`, it returns the previously created logger instance instead of creating a new one. This ensures consistent configuration and avoids duplicate handlers.

  > **Note**: Because `all_loggers` is a module-level global variable, it is **not shared across processes**. Each process maintains its own copy of `all_loggers`. Therefore, loggers are **isolated between processes**, which is the expected behavior in multiprocessing environments.

- **Default Log File Path**:  
  If `log_file` is not provided during the first call to `get_logger` for a given `logger_name`, the system attempts to determine a default log file path as follows:

  ```python
  default_log_dir = os.environ.get("QIUPUTILS_DEFAULT_LOG_DIR")
  if default_log_dir is None:
      raise ValueError(
          "Environment variable 'QIUPUTILS_DEFAULT_LOG_DIR' is not set. "
          "Please specify a log file explicitly via the 'log_file' argument, "
          "or set the 'QIUPUTILS_DEFAULT_LOG_DIR' environment variable."
      )
  log_file = os.path.join(default_log_dir, f"{logger_name}_{os.getpid()}.log")
  ```
  
  - The log file name includes the `logger_name` and the current process ID (`os.getpid()`) to ensure uniqueness across processes.
  - If the environment variable `QIUPUTILS_DEFAULT_LOG_DIR` is unset or evaluates to `None`, the function raises a `ValueError` to avoid silent failures or unintended log locations.
