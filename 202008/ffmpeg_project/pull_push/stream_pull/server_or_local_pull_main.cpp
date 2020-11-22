/**
 * Server or local pull stream
 * https://blog.csdn.net/zhuweigangzwg/article/details/43734251
 */
 
#include "server_or_local_pull.hpp"

int main(int argc ,char ** argv)  
{  
    InitInput(INPUTURL,&icodec,&ifmt);  
    InitOutput();  
    printf("--------程序运行开始----------\n");  
    //////////////////////////////////////////////////////////////////////////  
    Es2Mux_2();  
    //////////////////////////////////////////////////////////////////////////  
    UintOutput();  
    UintInput();  
    printf("--------程序运行结束----------\n");  
    printf("-------请按任意键退出---------\n");  
    return getchar();  
}
