#!/usr/bin/env python

'''
Measuring python opencv API
'''
# python3
from __future__ import print_function

# dirctory
from glob import glob
import cv2 as cv
# regular expression
import re

if __name__ == '__main__':
    cv2_callback = set(['cv.' + name for name in dir(cv) if callable(getattr(cv ,name))])
    found = set()
    for fn in glob('*.py'):
        print(' ----- ',fn)
        code = open(fn).read()
        found |= set(re.findall('cv2?\.\w+' ,code))

    cv2_used = found & cv2_callback
    cv2_unused = cv2_callback - cv2_used
    with open('unused_api.txt' ,'w') as f:
        f.write('\n'.join(sorted(cv2_unused)))
    
    r = 1.0 * len(cv2_used) / len(cv2_callback)
    print('\ncv api coverage: %d / %d (%.1f%%)' % (len(cv2_used) ,len(cv2_callback) ,r*100))
