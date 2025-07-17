
#include    "include.h"
#ifndef     _DRVSYSTEM_H
#define     _DRVSYSTEM_H

typedef enum
{
    LVDSRX  = 0x00,     
    MIPIRX  = 0x01,     //pcr recover video timing
    TTLRX   = 0x02,
    PATTERN = 0x03,
    LVDSDEBUG = 0x04,
    MIPIDEBUG = 0x05,
    TTLDEBUG  = 0x06,
    
}EnumChipRxSrc;

typedef enum
{
    LVDSTX  = 0x00,     
    MIPITX  = 0x01,
    
}EnumChipTxSel;

typedef enum
{
    AD_MLTX_READ_CLK    = 0x08,   //0x08
    AD_MLTX_WRITE_CLK   = 0x09,   //0x09
    AD_DESSCPLL_PIX_CLK = 0x10,   //0x10
    AD_RXPLL_PIX_CLK    = 0x1a,   //0x1a
    AD_DESSCPLL_PCR_CLK = 0x14,   //0x14
    AD_MLRXA_BYTE_CLK   = 0x18,
    AD_MLRXB_BYTE_CLK   = 0x1e,
}Enum_FM_CLK;


typedef  enum
{
    HTOTAL_POS    =    0,
    HACTIVE_POS,
    HFP_POS,
    HSW_POS,
    HBP_POS,
    
    VTOTAL_POS,
    VACTIVE_POS,
    VFP_POS,
    VSW_POS,
    VBP_POS,
    
    HSPOL_POS,
    VSPOL_POS,
}POS_INDEX;

typedef enum
{
    RXPLL_PIX_CLK     = 0x00,
    DESSCPLL_PIX_CLK  = 0x01,
    RXPLL_DEC_DDR_CLK = 0x02,
    MLRX_BYTE_CLK     = 0x03,
    
}Enum_VIDCHK_PIXCLK_SRC_SEL;

extern void Drv_SystemActRx_Sel(IN u8 ucSrc);
extern void Drv_SystemTxSram_Sel(IN u8 ucSrc);
extern u8 Drv_System_GetPixelEncoding(void);
extern void Drv_System_VidChk_SrcSel(u8 ucSrc);
extern void Drv_System_VidChkClk_SrcSel(u8 ucSrc);
extern u8 Drv_VidChk_FrmRt_Get(void);
extern u32 Drv_System_FmClkGet(IN u8 ucSrc);
extern void Drv_VidChkAll_Get(OUT StructVidChkTiming *video_time);

#endif