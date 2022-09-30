import os

def EnvCmdText():
    text = open('a' ,mode='r+' ,encoding='utf-8')
    out_text = open('b' ,mode='w+' ,encoding='utf-8')

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

if __name__ == "__main__":
    EnvCmdText()