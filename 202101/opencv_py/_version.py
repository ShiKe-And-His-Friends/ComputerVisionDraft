#!/usr/bin/env python

'''
prints OpenCv version
usage: 
    opencv_version.py [<params>]
    params:
        --build:print complete build info
        --help:print this help
'''

from __future__ import print_function
import numpy as np
import cv2 as cv

def main():
    import sys
    try:
        param = sys.argv[1]
    except IndexError:
        param = ""

    if "--build" == param:
        print(cv.getBuildInformation())
    elif "--help" == param:
        print("\t--build\n\t\tprint complete build info")
        print("\t--help\n\t\tprint this help")
    else:
        print("Welcome to OpenCV")
    print("DONE")

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
