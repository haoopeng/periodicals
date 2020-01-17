#!/usr/bin/env python
# encoding: utf-8
"""
MONitor & EXecute
Yong-Yeol Ahn (http://yongyeol.com/)

This script monitors multiple files and executes the given command when
any of those files is changed.
"""

import sys
import os
import time
from glob import glob
from optparse import OptionParser

def do(command):
    os.system("%s" % (command))
    print('done.', time.strftime('%X %x %Z'))

def get_all_files():
    ignore_keywords = ['.git', '.svn', '.hg', '.cache', '__pycache__']
    files = set()
    for root, dirnames, filenames in os.walk('./'):
        if any(1 for kw in ignore_keywords if kw in root):
            continue
        for filename in filenames:
            files.add(os.path.join(root, filename))
    return files

def get_filelist(args):
    if not args:
        return sorted(get_all_files())

    files = set()
    for arg in args:
        files |= set(glob(arg))

    return sorted(files)

def get_timestamp(files):
    return [os.stat(x).st_mtime for x in files]

def loop(old_timestamps):
    while(True):
        files = get_filelist(args)
        curr_timestamps = get_timestamp(files)
        if len(curr_timestamps) != len(old_timestamps) or \
           any(curr != old for curr, old in zip(curr_timestamps, old_timestamps)):
            do(command)
            old_timestamps = curr_timestamps
        time.sleep(2)

if __name__=='__main__':
    usage = 'usage: %prog -c "command" file1 file2 ...'
    parser = OptionParser(usage=usage)
    parser.add_option("-c", "--command", dest='command',
                      help="command to be executed")

    (options, args) = parser.parse_args()
    if not options.command:
        parser.error('-c option is needed.')

    command = options.command.strip('"')
    files = get_filelist(args)
    old_timestamps = get_timestamp(files)
    print('monitoring the following files...')
    print('\n'.join(files))
    time.sleep(2)

    do(command)
    loop(old_timestamps)
