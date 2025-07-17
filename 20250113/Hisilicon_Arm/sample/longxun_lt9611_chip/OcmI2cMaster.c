#include "include.h"




static unsigned char g_ucLogLevel = LOG_DEBUG;
static char *messageTypeStr[] = {"DEBUG","INFO","WARN", "ERROR","CRIT"};

//void Ocm_PrintLevel_Set(unsigned char ucLvl)
//{
//    g_ucLogLevel = ucLvl;
//}

void LTLog(unsigned char ucLvl, const char *fmt, ...)
{
    char buf[MAX_NUMBER_BYTES] = {0};
    va_list args;
    if (ucLvl > LOG_NOTRACE)
    {
        return;
    }
    
    
    if(ucLvl >= g_ucLogLevel)
    {
        va_start(args, fmt);
        vsprintf(buf, fmt, args);
        va_end(args);
        
        printf("\n[%-5s] %s", messageTypeStr[ucLvl], buf);
    }
}

