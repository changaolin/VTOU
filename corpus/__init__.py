from .handle import preText
from .utils import tagText,savePkl,test_input
__version__ = '0.0.1'
VERSION = tuple(map(int, __version__.split('.')))

__all__ = [
    'preText','tagText','savePkl','test_input'
]
