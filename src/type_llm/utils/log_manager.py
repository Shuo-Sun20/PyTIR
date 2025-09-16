import os
import logging
import inspect
from .config import logPath
import sys
from datetime import datetime

def setup_logger(log_dir = logPath, with_console = False):
    # 获取调用者的文件名 
    caller_frame = inspect.stack()[1]
    caller_filepath = caller_frame.filename
    caller_filename = os.path.splitext(os.path.basename(caller_filepath))[0]

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 创建或获取logger实例
    logger = logging.getLogger(caller_filename)
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 避免重复添加handler
    if not logger.handlers:
        # 创建文件handler
        time_str = datetime.now().strftime('%Y%m%d%H%M%S')
        log_file = os.path.join(log_dir, f'{caller_filename}_{time_str}.log')
        if os.path.exists(log_file):
            os.unlink(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # 设置写入文件的日志级别

        # 定义日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # 将handler添加到logger
        logger.addHandler(file_handler)
        
    if with_console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger