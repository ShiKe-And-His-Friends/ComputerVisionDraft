/******************************************************************************
  * @project: LT9211C
  * @file: ModSystem.h
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#ifndef _MODSYSTEM_H
#define _MODSYSTEM_H

typedef enum
{  
    STATE_CHIPRX_POWER_ON = 1,      //0x01
    STATE_CHIPRX_WAIT_SOURCE,      //0x02
    STATE_CHIPRX_VIDTIMING_CONFIG,  //0x03      
    STATE_CHIPRX_PLL_CONFIG,        //0x04 
    STATE_CHIPRX_VIDEO_CHECK,       //0x05
    STATE_CHIPRX_PLAY_BACK          //0x06
}EnumChipRxState;

typedef enum
{
    STATE_CHIPTX_POWER_ON = 1,          //0x01
    STATE_CHIPTX_UPSTREAM_VIDEO_READY,  //0x02
    STATE_CHIPTX_CONFIG_VIDEO,          //0x03
    STATE_CHIPTX_VIDEO_OUT,             //0x04
    STATE_CHIPTX_PLAY_BACK              //0x05
}EnumChipTxState;

typedef enum
{
    MIPIRX_VIDEO_ON_EVENT = 1,
    MIPIRX_VIDEO_OFF_EVENT,
    MIPIRX_CSC_EVENT,
}EnumChipRxEvent;

typedef struct ChipRx
{
    u8 b1RxStateChanged;
    u8 b1VidChkScanFlg;
    u8 ucPixelEncoding;
    u8 ucRxFormat;
    u8 ucRxState;
    void (*pHdmiRxNotify)(EnumChipRxEvent ucEvent); 
}StructChipRx;


typedef struct ChipTx
{
    u8 b1TxStateChanged;
    u8 b1UpstreamVideoReady;
    u8 ucTxState;
}StructChipTx;


extern StructChipRx g_stChipRx;
extern StructChipTx g_stChipTx;
extern StructVidChkTiming g_stVidChk;
extern StructChipRxVidTiming g_stRxVidTiming;



extern void Mod_System_RxNotifyHandle(EnumChipRxEvent ucEvent);
extern void Mod_SystemRx_NotifyRegister(void (*pFunction)());
extern void Mod_SystemTx_PowerOnInit(void);
extern void Mod_SystemRx_PowerOnInit(void);
extern void Mod_SystemRx_SetState(u8 ucState);
extern void Mod_SystemTx_SetState(u8 ucState);
extern void Mod_ChipID_Read(void);
extern void Mod_System_Init(void);
extern void Mod_LT9211C_Reset(void);

#endif