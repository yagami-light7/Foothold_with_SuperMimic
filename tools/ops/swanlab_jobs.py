"""Environment options for independent SwanLab tmux jobs."""
import os


PROXY_VARIABLES = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                   'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')


def tmux_proxy_options(environ=None):
    # tmux retains the server's old environment across SSH sessions. Override
    # every proxy variable, including removed ones, only in the new session.
    # Do not print or persist these options: a proxy URL can contain credentials.
    environ = os.environ if environ is None else environ
    return [item for key in PROXY_VARIABLES for item in ('-e', key+'='+environ.get(key, ''))]
