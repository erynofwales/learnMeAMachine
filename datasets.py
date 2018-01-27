#!/usr/bin/env python3

import hashlib
import os
import pickle
import requests
import tarfile

def requires_scratch(f):
    '''Decorator that ensures the scratch dir exists before calling the decorated function.'''
    def _wrapped(*args, **kwargs):
        _ensure_scratch_path()
        return f(*args, **kwargs)
    _wrapped.__name__ = f.__name__
    _wrapped.__doc__ = f.__doc__
    return _wrapped

#
# Dataset getters
#

class CIFAR10:
    def __init__(self, path):
        self.path = path

    @property
    def data_batch1(self):
        return self._do_data_batch(1)

    @property
    def data_batch2(self):
        return self._do_data_batch(2)

    @property
    def data_batch3(self):
        return self._do_data_batch(3)

    @property
    def data_batch4(self):
        return self._do_data_batch(4)

    @property
    def data_batch5(self):
        return self._do_data_batch(5)

    @property
    def test_batch(self):
        if not getattr(self, '__test_batch', None):
            path = os.path.join(self.path, 'test_batch')
            self.__test_batch = self._unpickle(path)
        return self.__test_batch

    @property
    def all_data_batches(self):
        return [getattr(self, 'data_batch{}'.format(i)) for i in range(1, 6)]

    def _do_data_batch(self, idx):
        attr = '__data_batch{}'.format(idx)
        if not getattr(self, attr, None):
            path = os.path.join(self.path, 'data_batch_{}'.format(idx))
            setattr(self, attr, self._unpickle(path))
        return getattr(self, attr)

    def _unpickle(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data


@requires_scratch
def cifar10():
    '''Download, extract, and return the CIFAR-10 archive.'''
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_md5 = 'c58f30108f718f92721af3b95e74349a'

    archive_path = os.path.join(_scratch_path(), 'cifar10.tar.gz')

    if not os.path.exists(archive_path):
        # Download the file to our scratch path.
        print('Downloading from {}'.format(url))
        r = requests.get(url)
        r.raise_for_status()
        with open(archive_path, 'wb') as f:
            print('Writing to {}'.format(archive_path))
            f.write(r.content)
    else:
        print('Archive exists, proceeding: {}'.format(archive_path))

    # TODO: Validate MD5 sum

    root = os.path.join(_scratch_path(), 'cifar-10-batches-py')
    if not os.path.isdir(root):
        with tarfile.open(archive_path) as f:
            print('Extracting')
            f.extractall(_scratch_path())

    return CIFAR10(root)

#
# Scratch helpers
#

def _ensure_scratch_path():
    try:
        os.makedirs(_scratch_path())
    except OSError as exc:
        if exc.errno != os.errno.EEXIST:
            raise

def _scratch_path():
    return os.path.join(_script_path(), 'scratch')

def _script_path():
    return os.path.dirname(os.path.abspath(__file__))
