"""Read members of a large zip on an FTP server without downloading it all, using byte-range requests."""
import io
import subprocess
import zipfile


class RemoteFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = subprocess.run(["curl", "-s", "--max-time", "600", "--retry", "5", "-r", f"{self.pos}-{self.pos + n - 1}", self.url], capture_output=True, check=True).stdout
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_remote(url, size, buffer=1 << 20):
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(url, size), buffer_size=buffer))
