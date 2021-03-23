#!/usr/bin/env python

'''
Scan *.py
report missing __doc__
'''

from __future__ import print_function
from glob import glob

if __name__  == '__main__':
    print('---- undocumented file:')
    for fn in glob('*.py'):
        loc = {}
        try:
            try:
                execfile(fn ,doc)
            except NameError:
                exec(open(fn).read() ,loc)
        except Exception:
            pass

        if '__doc__' not in loc:
            print(fn)

