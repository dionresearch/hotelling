"""helpers.py."""
from io import BytesIO
from warnings import warn

try:
    import sixel
except ImportError:
    sixel = None


def savefig(plt):
    """savefig.

    Allows displaying a matplotlib figure to the console terminal. This requires `pysixel` to be pip installed.
    It also requires a terminal with `Sixel graphic` support, like `DEC` with graphic support, Linux `xterm` (started
    with -ti 340), MLTerm (multilingual terminal, available on Windows, Linux etc).

    This is called by the command line stem tool when using -o stdout and can also be used in an ipython session.

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
