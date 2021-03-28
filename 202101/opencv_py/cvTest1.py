import __future__ as print_function
import cv2 as cv

def main():
    import sys
    var = sys.argv[1]
    src = cv.imread(var ,100 ,100)
    cv.imshow("print" ,src)
    cv.namedWindow("print" ,src)
    cv.waitKey(0)
    print("DONE")

if __name__ == "__main__":
    main()
