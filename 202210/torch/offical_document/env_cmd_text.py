import os

def EnvCmdText():
    text = open('a.txt' ,mode='r+' ,encoding='utf-8')
    out_text = open('b.txt' ,mode='w+' ,encoding='utf-8')

    i = 1
    lines = text.readlines()
    for line in lines:
        if i %2 == 0 :
            print(line.strip('\n'))
            out_text.writelines(line)
        i+=1
    text.close()
    out_text.close()

    return None