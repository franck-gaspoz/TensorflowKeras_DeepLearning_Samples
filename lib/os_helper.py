"""
OS helper functions
"""

import os


def print_os_env():
    """
    print os variable environments
    """
    for k, v in os.environ.items():
        print(k, ' = ', v)

