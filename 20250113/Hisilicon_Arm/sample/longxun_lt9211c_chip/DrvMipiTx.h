

#include	"include.h"

#ifndef		_DRVMIPITX_H
#define		_DRVMIPITX_H

typedef enum
{
    RGB_6Bit = 0,
    RGB_6Bit_L,
    RGB_565Bit,
    RGB_8Bit,
    RGB_10Bit,
    RGB_12Bit,
    YUV422_8bit,
    YUV444_8bit,
    YUV422_10bit,
    YUV422_12bit,
    YUV420_8bit,
    YUV420_10bit,
    MIPITX_FORMAT_CNT,
}Enum_MIPI_FORMAT;

typedef enum
{
    SPort  = 0x01, // Single port : 1port
    DPorts = 0x02,// Dual ports : 2port
}Enum_PORT_NUM;


typedef enum
{

    MIPITX_1LANE = 1,
    MIPITX_2LANE = 2,
    MIPITX_3LANE = 3,
    MIPITX_4LANE = 4,
    MIPITX_8LANE = 8,
}Enum_PORTLANE_NUM;


typedef enum
{
    NONE_3D = 0x00, //no 3d 
    LR_MODE = 0x10, //left/right
    TD_MODE = 0x20, //top/down
    OE_MODE = 0x40, //odd/even        
}Enum_3D_MODE;


typedef enum
{
    HALF_WRITE_CLK = 0,
    WRITE_CLK,
}EnumWrClkSel;

typedef struct MipiTx
{
    u8 b1MipiClockburst;
    u8 b1DphyCsi8Lane;
    u8 ucTxFormat;    
    u8 ucBpp;
    u8 ucTxPortNum;
    u8 ucTxLaneNum;
    u32 ulMipiDataRate;
    u32 ulMipiInClk;
    u32 ulMipiByteClk;
//    void (*pMipiTxNotify)(EnumMipiTxEvent ucEvent);
}StructMipiTx;

//PHY timing :
typedef struct MipiTxDPhy
{
    u8 ucClkZero;      
    u8 ucClkPre;       
    u8 ucClkPost;      
    u8 ucHsLpx;        
    u8 ucHsPrep;       
    u8 ucHsTrail;          
    u8 ucHsRqStPre;
}StructMipiTxDPhy;

extern StructMipiTx g_stMipiTx;
extern StructMipiTxDPhy g_stMipiTxDPhy;
extern void Drv_MipiTx_CsiDataTypeSet(IN u8 ucTxFormat);
extern void Drv_MipiTx_CsiFrameCntSet(void);
extern void Drv_MipiTx_DcsAutoMode(void);
extern void Drv_MipiTx_DPhyClkHsTrig(void);
extern void Drv_MipiTx_DPHYClkMode_Sel(IN u8 b1IsMipiClockburst);
extern void Drv_MipiTx_DPHYClkData_Set(void);
extern void Drv_MipiTx_DPHYCSI8Lane_En(IN u8 b1IsEn);
extern void Drv_MipiTx_DPhySet(void);
extern void Drv_MipiTx_DPhySkewCali(void);
extern void Drv_MipiTx_DsiDataTypeSet(IN u8 ucTxFormat);
extern void Drv_MipiTx_FifoDelay_Set(IN u16 rddly);
extern u16 Drv_MipiTx_FSMHact_Get(void);
extern u32 Drv_MipiTx_GetMipiInClk(INOUT StructMipiTx* pstMipiTx);
extern void Drv_MipiTx_HalfWrClkSrc_Sel(IN u8 clock);
extern u16 Drv_MipiTx_Hss_Get(void);
extern void Drv_MipiTx_Hss_Set(IN u16 value);
extern void Drv_MipiTx_InHSyncPol_Sel(IN u8 b1SyncPol);
extern void Drv_MipiTx_InVSyncPol_Sel(IN u8 b1SyncPol);
extern void Drv_MipiTx_LaneSet(IN u8 LaneNum);
extern void Drv_MipiTx_PhyTimingParaSet(INOUT StructMipiTx* pstMipiTx, INOUT StructMipiTxDPhy* pstMipiTxDPhy);
extern void Drv_MipiTx_PllSet(IN StructMipiTx* pstMipiTx);
extern void Drv_MipiTx_PortDataEnable(void);
extern void Drv_MipiTx_PortSet(IN u8 ucTxPortNum);
extern void Drv_MipiTx_PortCopy();
extern void Drv_MipiTx_TimingSet(IN StructVidChkTiming* pstVidTiming);
extern void Drv_MipiTx_VideoSet(u8 b1Opt);
extern u8 Drv_MipiTx_PllCali(void);

#endif