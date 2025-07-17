#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
*  Copyright (C), 2006-2022, Lontium Tech.
*  Project       : LT2103
*  File Name     : ModMipiTx.c
*  Version       : V1.0
*  Author        : sxue
*  Created       : 2022/7/15
*  Description   : 
*  
*  History:
*  2022/7/15     sxue      Created File
******************************************************************************/
#include    "include.h"

#if ((LT9211C_MODE_SEL == LVDS_IN_MIPI_OUT)||(LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)||(LT9211C_MODE_SEL == TTL_IN_MIPI_OUT))

StructMipiTx g_stMipiTx;
StructMipiTxDPhy g_stMipiTxDPhy;

static char* g_szStrTxFormat[MIPITX_FORMAT_CNT] = 
{
    "RGB 6bit",
    "RGB 6bit L",
    "RGB 565bit",
    "RGB 8bit",
    "RGB 10bit",
    "RGB 12bit",
    "YUV422 8bit",
    "YUV422 10bit",
    "YUV422 12bit",
    "YUV420 8bit",
    "YUV420 10bit",
};

//========================================================================
// Func Name   : Mod_SystemTx_PowerOnInit
// Description : mipi tx struct init
// Input       : void  
// Output      : None
// Return      : void
//========================================================================



void Mod_MipiTx_PortLane_Adj(void)
{
#if (MIPITX_OUT_SEL == MIPI_CSI)
    if (g_stMipiTx.ulMipiInClk > 600000 )//over 4K60
    {
        g_stMipiTx.b1DphyCsi8Lane = ENABLED;
        g_stMipiTx.ucTxPortNum = SPort;
        g_stMipiTx.ucTxLaneNum = MIPITX_8LANE;
    }
//    else
//    {
//        g_stMipiTx.b1DphyCsi8Lane = DISABLED;
//        g_stMipiTx.ucTxPortNum = SPort;
//        g_stMipiTx.ucTxLaneNum = MIPITX_4LANE;
//    }
#endif
}

//========================================================================
// Func Name   : Mod_MipiTx_DataRateAdj
// Description : calc DPHY data rate, limit up && low
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Mod_MipiTx_DataRateAdj(void)
{
    //MIPI D-PHY
    //DPHY is 8bit to 8bit
    g_stMipiTx.ulMipiDataRate = g_stMipiTx.ulMipiInClk * g_stMipiTx.ucBpp * 2 / (g_stMipiTx.ucTxPortNum * g_stMipiTx.ucTxLaneNum);

    //clk����ʱ���ӿ�80M��clk������ʱ���ӿ�120M�����˫port�ټӿ�40M��������DPHY timing LP��HS���л�������
    if(g_stMipiTx.b1MipiClockburst)
    {
        g_stMipiTx.ulMipiDataRate += 120000;
    }
    else
    {
        g_stMipiTx.ulMipiDataRate += 80000;
    }


    if(g_stMipiTx.ulMipiDataRate < MIPITX_PLL_LOW) //set byteclk minium value to 50M , phyclk minium value is 80M 
    {
        g_stMipiTx.ulMipiDataRate = MIPITX_PLL_LOW ;
    }
    if(g_stMipiTx.ulMipiDataRate > MIPITX_PLL_HIGH)//set byteclk maxmum value to 312.5M , phyclk maxmum value is 2.5G 
    {
        g_stMipiTx.ulMipiDataRate = MIPITX_PLL_HIGH ;
    }
}

void Mod_MipiTxVidFmt_Get()
{
    g_stMipiTx.ucTxFormat = MIPI_TX_FORMAT;
}

//========================================================================
// Func Name   : Mod_MipiTx_ParaSet
// Description : ���� port, lane, clockburst, 3dmode
//               get MipiInClk
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Mod_MipiTx_ParaSet(void)
{
    u32 ulTmpClk ;

    Mod_MipiTxVidFmt_Get();
    g_stMipiTx.ucTxPortNum = SPort;
    g_stMipiTx.ucTxLaneNum = MIPITX_LANE_NUM;
    g_stMipiTx.b1MipiClockburst = MIPI_CLOCK_BURST;
    printf("\nburst:0x%02x", g_stMipiTx.b1MipiClockburst);

    switch (g_stMipiTx.ucTxFormat)
    {
        case RGB_6Bit:
            g_stMipiTx.ucBpp = 18;
            break;
        case RGB_6Bit_L: //bpp24
           g_stMipiTx.ucBpp = 24;
            break;
        case RGB_565Bit: //bpp216
           g_stMipiTx.ucBpp = 16;
            break;
        case RGB_8Bit:
            g_stMipiTx.ucBpp = 24;
            break;
        case YUV420_8bit:
            g_stMipiTx.ucBpp = 24;
            break;
        case RGB_10Bit:
        case YUV420_10bit:
            g_stMipiTx.ucBpp = 30;
            break;
        case RGB_12Bit:
            g_stMipiTx.ucBpp = 36;
            break;
        case YUV422_8bit:
            g_stMipiTx.ucBpp = 16;
            break;
        case YUV422_10bit:
            g_stMipiTx.ucBpp = 20;
            break;
        case YUV422_12bit:
            g_stMipiTx.ucBpp = 24;
            break;
        
        default:
            g_stMipiTx.ucBpp = 24;
            break;
    }

    Drv_MipiTx_GetMipiInClk(&g_stMipiTx);

    //MIPI D-PHY check either use 2port or not
    ulTmpClk = g_stMipiTx.ulMipiInClk * g_stMipiTx.ucBpp * 2 / (g_stMipiTx.ucTxLaneNum);
    if (ulTmpClk > MIPITX_PLL_HIGH)
    {
        printf("\nOver Maximum MIPITX Datarate");
    }

    //csi port adjust
//    Mod_MipiTx_PortLane_Adj();
}

//========================================================================
// Func Name   : Mod_MipiTx_Resolution_Config
// Description : get vid chk timing, set mipi resolution config
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Mod_MipiTx_Resolution_Config(void)
{
    Drv_VidChkAll_Get(&g_stVidChk);
    Drv_MipiTx_TimingSet(&g_stVidChk);
    Drv_MipiTx_InHSyncPol_Sel(g_stVidChk.ucHspol);
    Drv_MipiTx_InVSyncPol_Sel(g_stVidChk.ucVspol);
}

void Mod_MipiTx_HssSet(void)
{
    u16 Hss;
    #if (MIPITX_OUT_SEL == MIPI_DSI)
    Hss = 0x0A; //Hss
    #else 
    if(g_stMipiTx.b1MipiClockburst == ENABLED)
    {
        Hss = (3 * g_stMipiTxDPhy.ucHsRqStPre + g_stMipiTxDPhy.ucHsTrail + 9) / 2 + 22; //Hss
    }
    else
    {
        Hss = (g_stMipiTxDPhy.ucHsRqStPre + (g_stMipiTxDPhy.ucHsTrail + 13) / 2) + 20;
    }    
    #endif
    Drv_MipiTx_Hss_Set(Hss);
}

void Mod_MipiTx_FifoRddly_Config(void)
{
    u16 ushss, usrgodhact;
    u16 ulRdHalfpixclk, usrdbyteclk;
    u32 ulTemp, ulrddly_max, ulrddly_min1, ulrddly_min2;
    u32 ulrddly = 0;

    //MIPI_DPHY Dphy is double-byte design , Cphy not
    ulRdHalfpixclk =  (u16)(g_stMipiTx.ulMipiInClk / 1000); //half pix
//    usrdbyteclk = (Drv_System_FmClkGet(AD_MLTX_WRITE_CLK) / 2); //ʹ��ʵ�ʵ�byteclk�����ƫС

    usrdbyteclk = (u16)(g_stMipiTx.ulMipiDataRate / 2000);
    usrdbyteclk = usrdbyteclk / 8;
  
    usrdbyteclk *= g_stMipiTx.ucTxPortNum;
    
    if((g_stMipiTx.ucTxLaneNum == 8)&&(g_stMipiTx.ucTxPortNum == 1))
    {
        usrdbyteclk <<= 1;
    }
    
    ushss = Drv_MipiTx_Hss_Get();
    usrgodhact = Drv_MipiTx_FSMHact_Get();
    ulTemp = (usrdbyteclk * ((g_stVidChk.usHs >> 1) + (g_stVidChk.usHbp >> 1)) / ulRdHalfpixclk); 
    
    if(ulTemp > ushss)
    {
        ulrddly_min1 = (usrdbyteclk * ((g_stVidChk.usHs >> 1) + (g_stVidChk.usHbp >> 1)) / ulRdHalfpixclk) - ushss;
    }
    else
    {
        ulrddly_min1 = 0;
    }
    
    if(g_stMipiTx.ucTxPortNum == 1)
    {
        ulrddly_min2 = (u32)usrdbyteclk * ((g_stVidChk.usHs >> 1) + (g_stVidChk.usHbp >> 1) + (g_stVidChk.usHact >> 1));
        ulTemp = ((u32)ulrddly_min2 / (u32)ulRdHalfpixclk);
        if(ulTemp > (ushss + usrgodhact))
        {
            ulrddly_min2 =(u32)(((u32)ulrddly_min2 / (u32)ulRdHalfpixclk) - ushss - usrgodhact);
        }
        else
        {
            ulrddly_min2 = 0;
        }
    }
    else
    {
        ulrddly_min2 = ulrddly_min1;
    }
    
    if (ulrddly_min1 > ulrddly_min2)
    {
        ulrddly_min2 = ulrddly_min1;
    }

    ulrddly_max = 0x8000 / (g_stMipiTx.ucBpp << 1);  //0x8000: 512fifo * 8 byte * 8 bit
    ulrddly_max = usrdbyteclk * (ulrddly_max + (g_stVidChk.usHs >> 1) + (g_stVidChk.usHbp >> 1));
    ulrddly_max = ulrddly_max / (u32)ulRdHalfpixclk - ushss;
//    ulrddly = (ulrddly_max - ulrddly_min2) / 20 + ulrddly_min2;
    ulrddly = (ulrddly_max / 7) + ulrddly_min2;

    Drv_MipiTx_FifoDelay_Set(ulrddly);
    printf("\nrddly is 0x%04lx;",ulrddly);          
}

//========================================================================
// Func Name   : Mod_MipiTx_Digital_Config
// Description : mipi tx digital config
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Mod_MipiTx_Digital_Config(void)
{
    Drv_MipiTx_LaneSet(g_stMipiTx.ucTxLaneNum);
    Drv_MipiTx_PortCopy();
    Drv_MipiTx_PortDataEnable();
    Drv_MipiTx_DPHYClkData_Set();
    Drv_MipiTx_DPHYClkMode_Sel(g_stMipiTx.b1MipiClockburst);
    if (g_stMipiTx.b1DphyCsi8Lane == ENABLED)
    {
        Drv_MipiTx_HalfWrClkSrc_Sel(WRITE_CLK);
        Drv_MipiTx_DPHYCSI8Lane_En(ENABLED);
    }
    Drv_MipiTx_PortSet(g_stMipiTx.ucTxPortNum);
    #if (MIPITX_OUT_SEL == MIPI_DSI)
    Drv_MipiTx_DsiDataTypeSet(g_stMipiTx.ucTxFormat);
    Drv_MipiTx_DcsAutoMode();
    #else
    Drv_MipiTx_CsiDataTypeSet(g_stMipiTx.ucTxFormat);
    Drv_MipiTx_CsiFrameCntSet();
    #endif
    printf("\nMipiTx Output Format: %s",g_szStrTxFormat[g_stMipiTx.ucTxFormat]);
}


void Mod_MipiTx_StateHandler(void)
{
    switch (g_stChipTx.ucTxState)
    {
        case STATE_CHIPTX_POWER_ON:
            memset(&g_stMipiTx, 0, sizeof(StructMipiTx));
            Mod_SystemTx_SetState(STATE_CHIPTX_UPSTREAM_VIDEO_READY);
            break;

        case STATE_CHIPTX_UPSTREAM_VIDEO_READY:
            if(g_stChipTx.b1TxStateChanged == true)
            {
                Drv_MipiTx_VideoSet(OFF);
                g_stChipTx.b1TxStateChanged = false;
            }
            
            if(g_stChipTx.b1UpstreamVideoReady == true)
            {
                Mod_SystemTx_SetState(STATE_CHIPTX_CONFIG_VIDEO);
            }
            break;
        case STATE_CHIPTX_CONFIG_VIDEO:
            Drv_SystemTxSram_Sel(MIPITX);
            Mod_MipiTx_ParaSet();
            Mod_MipiTx_DataRateAdj();
            
            Drv_MipiTx_PllSet(&g_stMipiTx);
            if (Drv_MipiTx_PllCali() == SUCCESS)
            {
                Drv_MipiTx_HalfWrClkSrc_Sel(HALF_WRITE_CLK);
                Mod_MipiTx_Resolution_Config();
                Drv_MipiTx_DPhySet();
                Drv_MipiTx_PhyTimingParaSet(&g_stMipiTx, &g_stMipiTxDPhy);
                Mod_MipiTx_Digital_Config();
                Mod_MipiTx_HssSet();
                Mod_MipiTx_FifoRddly_Config();

                Drv_MipiTx_DPhyClkHsTrig();
                Drv_MipiTx_DPhySkewCali();

                Mod_SystemTx_SetState(STATE_CHIPTX_VIDEO_OUT);
            }
            
            break;
        case STATE_CHIPTX_VIDEO_OUT:
			Drv_MipiTx_PanelInit();
            
            Drv_MipiTx_VideoSet(ON);
            
            Mod_SystemTx_SetState(STATE_CHIPTX_PLAY_BACK);
            break;
        case STATE_CHIPTX_PLAY_BACK:
            ;
        break;    
    }
}

void Mod_MipiTx_StateJudge(void)
{
    //monitor upstream video stable.
    if(g_stChipTx.ucTxState > STATE_CHIPTX_UPSTREAM_VIDEO_READY)
    {
        if(g_stChipTx.b1UpstreamVideoReady == false)
        {
            Drv_MipiTx_VideoSet(OFF);
            Mod_SystemTx_SetState(STATE_CHIPTX_UPSTREAM_VIDEO_READY);
        }
    }
}

void Mod_MipiTx_Handler(void)
{   
    Mod_MipiTx_StateJudge();
    Mod_MipiTx_StateHandler();
}


#endif
