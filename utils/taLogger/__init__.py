from .taLogger import taLogging
# from worker.redisWorker import redisWorker
#
#
__version__ = '0.0.1'
VERSION = tuple(map(int, __version__.split('.')))

__all__ = [
    'taLogging'
]