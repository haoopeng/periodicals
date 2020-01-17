""" doit script for journal2vec project

Data preprocessing:
- task_download_rawdata()

"""

#!/usr/bin/env python
# encoding: utf-8
import os
from doit.tools import run_once

def task_download_rawdata():
    """ download the raw data from server """
    remote_path = 'snowball:/l/nx/data/MicrosoftAcademicGraph/'
    local_path = 'data/raw_data/'

    # files = ['Journals.txt',
    #          'Conferences.txt',
    #          'PaperAuthorAffiliations.txt',
    #          'PaperReferences.txt',
    #          'Papers.txt',
    #          'readme.txt',
    #          'license.txt']

    files = ['Journals.txt',
             'Conferences.txt',
             'PaperAuthorAffiliations.txt',
             'PaperReferences_J_C.csv',
             'Papers_J_C.csv',
             'readme.txt',
             'license.txt']


    def download_all():
        """ download every file listed in files using scp """
        for fname in files:
            os.system('scp {} {}'.format(os.path.join(remote_path, fname),
                                         os.path.join(local_path, fname)))

    return {
        'actions': [download_all],
        'targets': [os.path.join(local_path, f) for f in files],
        'uptodate': [run_once],
    }
