

typedef enum {
  ERROR = 0,
  SUCCESS = !ERROR
}
ErrorStatus;


typedef enum
{
  OFF = 0,
  ON  = 1
}
Fun_Status;

typedef  enum
{
	Low    = 0,
	High   = !Low
}

Pin_Status;
typedef enum {FALSE = 0, TRUE = !FALSE} PinStatus;
typedef enum {RESET = 0, SET = !RESET} FlagStatus, ITStatus, BitStatus, BitAction;



typedef enum {
  Disable = 0,
  Enable = !Disable
}
FunctionalState;

#ifndef min
#define min(a,b) (((a)<(b))? (a):(b))
#endif

#ifndef max
#define max(a,b) (((a)>(b))? (a):(b))
#endif

#define     CID_READ				0x0B
#define     DID_READ				0x0C

#define     ERASE_APROM				0x22
#define     READ_APROM				0x00
#define     PROGRAM_APROM			0x21
#define     ERASE_LDROM				
#define     READ_LDROM				
#define     PROGRAM_LDROM			
#define     READ_CFG					0xC0
#define     PROGRAM_CFG				0xE1
#define			READ_UID					0x04


