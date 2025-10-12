"""
简单的文件锁实现
"""

import os
import time
import fcntl
import errno
from contextlib import contextmanager

class FileLock:
    """简单的文件锁类"""
    
    def __init__(self, lock_file, timeout=10):
        self.lock_file = lock_file
        self.timeout = timeout
        self.fd = None
    
    def acquire(self):
        """获取锁"""
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return True
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if time.time() - start_time > self.timeout:
                    return False
                time.sleep(0.1)
    
    def release(self):
        """释放锁"""
        if self.fd is not None:
            os.close(self.fd)
            try:
                os.unlink(self.lock_file)
            except OSError:
                pass
            self.fd = None
    
    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"无法获取锁: {self.lock_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

@contextmanager
def file_lock(lock_file, timeout=10):
    """文件锁上下文管理器"""
    lock = FileLock(lock_file, timeout)
    try:
        lock.acquire()
        yield lock
    finally:
        lock.release()


