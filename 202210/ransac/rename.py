import os
import re
import sys
def renameall():
    fileList = os.listdir(r".")       #待修改文件夹
    for fileName in fileList:       #遍历文件夹中所有文件
        dir_name =  str(fileName).split("-")
        print(dir_name)     #输出文件夹中包含的文件
        '''
        pat=".+\.(Bmp|bmp|jpg|png)"      #匹配文件名正则表达式
        pattern = re.findall(pat,fileName)      #进行匹配
        print(pattern)
        '''
        os.rename(fileName,(dir_name[-1]))       #文件重新命名
        
    print("---------------------------------------------------")
    

renameall()