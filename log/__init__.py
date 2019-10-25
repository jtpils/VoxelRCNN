import logging
from config import cfg
from datetime import datetime


def setup_logger(name, log_file, formatter, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class GlobalLogger(object):
    """Use a global logger to easily document all infos"""
    def __init__(self, level):
        super(GlobalLogger, self).__init__()
        self.name = "Global_log"
        self.level = level
        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.logger = setup_logger(self.name,
                             cfg.log.root,
                             self.formatter,
                             level=self.level)
        
    
    def header(self):
        """The header for each log
        Content:
        0. Split line
        1. Date and time
        2. Restart or Not
        3. Which GPUs
        etc.
        """
        self.head_formatter = logging.Formatter('%(message)s')
        self.head_logger = setup_logger("Header",
                                        cfg.log.root,
                                        self.head_formatter,
                                        level=logging.INFO)
        self.head_logger.info("\n\n" + "*" * 30)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.head_logger.info(current_time)
        if cfg.model.restart:
            self.head_logger.info("Using past training on: {}".format(cfg.model.savepath)) 
        self.head_logger.info("Using GPU: {}".format(cfg.CUDA_VISIBLE_DEVICES))
        self.head_logger.info("-" * 30)

    def info(self, input):
        self.logger.info(input)

    def warning(self, input):
        self.logger.warning(input)

if __name__ == "__main__":
    # folder_path = "./log/test"
    # tensorboard_folder_log(folder_path)
    pass