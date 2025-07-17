#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
  * @project: LT9211C
  * @file: DrvSystem.c
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#include "include.h"

#if LT9211C_MODE_SEL != PATTERN_OUT

void Drv_SystemActRx_Sel(IN u8 ucSrc)
{
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) & 0xf8));

    switch(ucSrc)
    {
        case LVDSRX:
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) | LVDSRX));
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) & 0xcf)); //[5:4]00: LVDSRX
        break;
        case MIPIRX:
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) | MIPIRX));
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) | BIT4_1)); //[5:4]01: MIPIRX
        break;
        case TTLRX:
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) | TTLRX));
        break;
        case PATTERN:
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) | PATTERN));
        break;
        default:
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) | LVDSRX));
        break;
        
    }
}

void Drv_SystemTxSram_Sel(IN u8 ucSrc)
{
    //[7:6]2'b00: TX Sram sel MIPITX; others sel LVDSTX
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) & 0x3f)); 

    switch(ucSrc)
    {
        case LVDSTX:
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) | BIT6_1));
        break;
        case MIPITX:
            HDMI_WriteI2C_Byte(0x30,(HDMI_ReadI2C_Byte(0x30) & BIT6_0));
        break;
    }
}

u8 Drv_System_GetPixelEncoding(void)
{
    return g_stChipRx.ucRxFormat;
}


void Drv_System_VidChkClk_SrcSel(u8 ucSrc)
{
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x80,(HDMI_ReadI2C_Byte(0x80) & 0xfc));

    switch (ucSrc)
    {
        case RXPLL_PIX_CLK:
            HDMI_WriteI2C_Byte(0x80,(HDMI_ReadI2C_Byte(0x80) | RXPLL_PIX_CLK));
        break;
        case DESSCPLL_PIX_CLK:
            HDMI_WriteI2C_Byte(0x80,(HDMI_ReadI2C_Byte(0x80) | DESSCPLL_PIX_CLK));
        break;
        case RXPLL_DEC_DDR_CLK:
            HDMI_WriteI2C_Byte(0x80,(HDMI_ReadI2C_Byte(0x80) | RXPLL_DEC_DDR_CLK));
        break;
        case MLRX_BYTE_CLK:
            HDMI_WriteI2C_Byte(0x80,(HDMI_ReadI2C_Byte(0x80) | MLRX_BYTE_CLK));
        break;    
        
    }

}

void Drv_System_VidChk_SrcSel(u8 ucSrc)
{
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0x3f,(HDMI_ReadI2C_Byte(0x80) & 0xf8));

    switch (ucSrc)
    {
        case LVDSRX:
            HDMI_WriteI2C_Byte(0x3f,LVDSRX);
        break;
        case MIPIRX:
            HDMI_WriteI2C_Byte(0x3f,MIPIRX);
        break;
        case TTLRX:
            HDMI_WriteI2C_Byte(0x3f,TTLRX);
        break;
        case PATTERN:
            HDMI_WriteI2C_Byte(0x3f,PATTERN);
        break;
        case LVDSDEBUG:
            HDMI_WriteI2C_Byte(0x3f,LVDSDEBUG);
        case MIPIDEBUG:
            HDMI_WriteI2C_Byte(0x3f,MIPIDEBUG);
        break;
        case TTLDEBUG:
            HDMI_WriteI2C_Byte(0x3f,TTLDEBUG);
        break;    
        
    }

}


u16 Drv_VidChkSingle_Get(u8 ucPara)
{ 
    u16 usRtn = 0;

    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x0b,0x7f);
    HDMI_WriteI2C_Byte(0x0b,0xff);
    usleep(80*1000);
    HDMI_WriteI2C_Byte(0xff,0x86);
    switch(ucPara)
    {
        case HTOTAL_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x60) << 8) + HDMI_ReadI2C_Byte(0x61);
        break;
        case HACTIVE_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x5c) << 8) + HDMI_ReadI2C_Byte(0x5d);  
        break;
        case HFP_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x58) << 8) + HDMI_ReadI2C_Byte(0x59);
        break;
        case HSW_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x50) << 8) + HDMI_ReadI2C_Byte(0x51);
        break;    
        case HBP_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x54) << 8) + HDMI_ReadI2C_Byte(0x55);
        break;
        case VTOTAL_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x62) << 8) + HDMI_ReadI2C_Byte(0x63);
        break;
        case VACTIVE_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x5e) << 8) + HDMI_ReadI2C_Byte(0x5f);
        break;
        case VFP_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x5a) << 8) + HDMI_ReadI2C_Byte(0x5b);
        break;
        case VSW_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x52) << 8) + HDMI_ReadI2C_Byte(0x53);
        break;
        case VBP_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x56) << 8) + HDMI_ReadI2C_Byte(0x57);
        break;
        case HSPOL_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x4f) & 0x01);
        break;
        case VSPOL_POS:
            usRtn = (HDMI_ReadI2C_Byte(0x4f) & 0x02);
        break;
        default:
        break;
    }
    return usRtn;
}

void Drv_VidChkAll_Get(OUT StructVidChkTiming *video_time)
{
    video_time->usHtotal    =     Drv_VidChkSingle_Get(HTOTAL_POS);
    video_time->usHact      =     Drv_VidChkSingle_Get(HACTIVE_POS);
    video_time->usHfp       =     Drv_VidChkSingle_Get(HFP_POS);
    video_time->usHs        =     Drv_VidChkSingle_Get(HSW_POS);
    video_time->usHbp       =     Drv_VidChkSingle_Get(HBP_POS);
    
    video_time->usVtotal    =     Drv_VidChkSingle_Get(VTOTAL_POS);
    video_time->usVact      =     Drv_VidChkSingle_Get(VACTIVE_POS);
    video_time->usVfp       =     Drv_VidChkSingle_Get(VFP_POS);
    video_time->usVs        =     Drv_VidChkSingle_Get(VSW_POS);
    video_time->usVbp       =     Drv_VidChkSingle_Get(VBP_POS);
    
    video_time->ucHspol     =     Drv_VidChkSingle_Get(HSPOL_POS);
    video_time->ucVspol     =     Drv_VidChkSingle_Get(VSPOL_POS);        
    video_time->ucFrameRate =     Drv_VidChk_FrmRt_Get(); 
}

#endif

u8 Drv_VidChk_FrmRt_Get(void)
{
    u8 ucframerate = 0; 
    u32 ulframetime = 0;

    HDMI_WriteI2C_Byte(0xff,0x86);
    ulframetime = HDMI_ReadI2C_Byte(0x43);
    ulframetime = (ulframetime << 8) + HDMI_ReadI2C_Byte(0x44);
    ulframetime = (ulframetime << 8) + HDMI_ReadI2C_Byte(0x45);
    ucframerate = (u8)(((float)25000000 / (float)(ulframetime)) + (float)(0.5)); //2500000/ulframetime
    return ucframerate;
}

u32 Drv_System_FmClkGet(IN u8 ucSrc)
{
    u32 ulRtn = 0;
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0X90,ucSrc);
    usleep(5);
    if ((HDMI_ReadI2C_Byte(0x98) & 0x60) == 0x60)
	{
		printf("\nFM CLOCK DET Stable");
	}
    else
	{
        printf("\nFrequency meter output clock status unStable");
	}
    
    HDMI_WriteI2C_Byte(0x90,(ucSrc | BIT7_1));
    ulRtn = (HDMI_ReadI2C_Byte(0x98) & 0x0f);
    ulRtn = (ulRtn << 8) + HDMI_ReadI2C_Byte(0x99);
    ulRtn = (ulRtn << 8) + HDMI_ReadI2C_Byte(0x9a);
    HDMI_WriteI2C_Byte(0x90,(HDMI_ReadI2C_Byte(0x90) & BIT7_0));
    return ulRtn;
}