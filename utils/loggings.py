import logging
class logger():
    def __init__(self, logger_name, log_path, mode, level=logging.INFO, format = "%(asctime)s - %(message)s"):
        self.logging = logging.getLogger(logger_name)
        self.logging.setLevel(level)
        fh = logging.FileHandler(log_path, mode)
        fh.setLevel(level)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter(format)
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        self.logging.addHandler(fh)
        self.logging.addHandler(sh)

    def info(self, msg):
        return self.logging.info(msg)


