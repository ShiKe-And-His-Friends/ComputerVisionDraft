#!/usr/bin/python3

import threading
import time

class myThread(threading.Thread):
    def __init__(self ,threadID ,name ,counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("start thread:" + self.name)
        threadLock.acquire()
        print_time(self.name ,self.counter ,3)
        threadLock.release()
        print("exit thread:" + self.name)

def print_time(threadName ,delay ,counter):
    while counter:
        time.sleep(delay)
        print("%s : %s" %(threadName ,time.ctime(time.time())))
        counter -= 1

threadLock = threading.Lock()
threads = []

thread1 = myThread(1 ,"Thread-1" ,1)
thread2 = myThread(2 ,"Thread-2" ,2)

thread1.start()
thread2.start()
threads.append(thread1)
threads.append(thread2)
for t in threads:
    t.join()
print("exit main thread")

