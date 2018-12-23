# https://www.cnblogs.com/Nicholas0707/p/9021672.html
import logging.config,logging
import os
import threading


class taLogger(object):
    _instance_lock = threading.Lock()
    _logger_dict = {}
    _conf_file = os.path.join(os.path.dirname(__file__), 'configs/config.conf')

    def load_config(self,file = _conf_file):
        logging.config.fileConfig(file)

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(taLogger,'_instance'):
            with taLogger._instance_lock:
                if not hasattr(taLogger, '_instance'):
                    taLogger._instance = object.__new__(cls)
        return taLogger._instance

    def _getLogger(self,name):
        logger = None
        try:
            if name in taLogger._logger_dict:
                logger = taLogger._logger_dict[name]
            else:
                logger = logging.getLogger(name)
                # rv = LoggerAdapter(logger, extra)
                taLogger._logger_dict[name] = logger

        finally:
            pass
        return logger

    def getFileLogger(self,name='fileligger',level=1,file='logs/fileligger.log'):
        logger = self._getLogger(name=name)
        logging.basicConfig()
        hander = logging.FileHandler(filename=file)
        format = logging.Formatter(fmt='%(asctime)s\t%(filename)s:\t\t%(lineno)d\t%(levelname)s\t%(name)s\t%(message)s',
                                   datefmt='[%Y-%m-%d %H:%M:%S]')
        hander.setFormatter(format)
        try:
            logger.handlers.pop()
        except Exception as e:
            pass
        logger.addHandler(hander)
        logger.setLevel(level*10)
        return logger
    pass

    def getConfLogger(self,file,logger,name='default'):
        self.load_config(file)
        logger = self._getLogger(name=name)
        return logger
        pass

taLogging =  taLogger()

if __name__ == '__main__':
    ta = taLogger()
    ta.load_config()
    logger = ta._getLogger(name='test')
    dic={
        "msg":123,
        "info":"你好"
    }
    logger.error(dic)

#     pass
