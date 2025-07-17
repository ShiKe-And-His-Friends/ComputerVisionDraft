#ifndef	  _OCMI2CMASTER_H_
#define	  _OCMI2CMASTER_H_

#define MAX_NUMBER_BYTES  128

typedef enum
{
    LOG_DEBUG =0x00,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_CRITICAL,
    LOG_NOTRACE,
} LT_LogLevel;

void LTLog(unsigned char ucLvl, const char *fmt, ...);


#endif
