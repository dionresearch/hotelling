"""helpers.py."""
from io import BytesIO
import os
from warnings import warn
import pandas as pd

try:
    import dask.dataframe as dd
except ImportError:
    dd = False
try:
    import sixel
except ImportError:
    sixel = None


def savefig(plt):
    """savefig.

    Allows displaying a matplotlib figure to the console terminal. This requires `pysixel` to be pip installed.
    It also requires a terminal with `Sixel graphic` support, like `DEC` with graphic support, Linux `xterm` (started
    with -ti 340), MLTerm (multilingual terminal, available on Windows, Linux etc).

    This is called by the command line tool when using --output stdout and can also be used in an ipython session.

    :param plt: matplotlib pyplot
    :return:
    """
    buf = BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    if sixel is None:
        warn("No sixel module available. Please install pysixel")
    writer = sixel.SixelWriter()
    writer.draw(buf)


def load_df(filepath, server=None, dask=None, **kwargs):
    """load_df.

    :param str filepath:
    :param str server: head node for distributed cluster, ip address and port or hostname and port (localhost for local)
    :param bool dask: if True, forces the use of dask,, even on smaller datasets
    :param kwargs: to pass arguments to pandas `read_csv`

    :return: dataframe
    """
    try:
        statinfo = os.stat(filepath)  # file could be on hdfs or s3
        filesize = statinfo.st_size
        if filesize > 2 * 1024 ** 3:  # 2GB, consider large
            large = True
        else:
            large = dask
    except (OSError, FileNotFoundError):
        # doesn't exist, or is distributed.
        filesize = 0
        large = True

    if server:
        large = True  # force it when server is specified

    set_index = None
    if dd and large:  # dask is available
        data_frame = dd
        if "index_col" in kwargs.keys():
            # for dask, we set index a different way, not in read_csv itself
            index_col = kwargs.pop("index_col")
            set_index = index_col
        if server:
            from distributed import Client

            if server == "localhost":
                client = Client()  # "distributed", local
            else:
                client = Client(server)  # distributed, head node
            print(client.ncores())
    else:
        data_frame = pd
    if set_index:
        df = data_frame.read_csv(filepath, **kwargs).set_index(index_col, sorted=True, drop=True)
    else:
        df = data_frame.read_csv(filepath, **kwargs)
    return df
