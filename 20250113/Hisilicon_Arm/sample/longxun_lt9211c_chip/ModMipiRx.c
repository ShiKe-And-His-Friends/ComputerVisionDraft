#include <stdio.h>
#include <stdlib.h>
/******************************************************************************
  * @project: LT9211C
  * @file: ModMipiRx.c
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/
#include    "include.h"


#if ((LT9211C_MODE_SEL == MIPI_IN_LVDS_OUT)||(LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)||(LT9211C_MODE_SEL == MIPI_IN_TTL_OUT))

void Mod_MipiRx_Init()
{
    memset(&g_stPcrPara,0 ,sizeof(StructPcrPara));
    memset(&g_stMipiRxVidTiming_Get,0 ,sizeof(SrtuctMipiRx_VidTiming_Get));
}

void Mod_MipiRxDig_Set()
{
    Drv_MipiRx_InputSel();
    Drv_MipiRx_LaneSet();
}

u8 Mod_MipiRx_VidChk_Stable()
{
    HDMI_WriteI2C_Byte(0xff, 0x86);
    if((HDMI_ReadI2C_Byte(0x40) & 0x01) == 0x01)
    {
        return true;
    }
    else
    {
        return false;
    }

}

//void Mod_MipiRx_Statejudge(void)
//{

//    if (g_stChipRx.ucRxState > STATE_CHIPRX_VIDTIMING_CONFIG)
//    {
//        if (Drv_MipiRx_VidFmtUpdate() == TRUE)
//        {
//            //LTLog(LOG_WARN,"Video Format Change");
//            g_stChipRx.pHdmiRxNotify(MIPIRX_VIDEO_OFF_EVENT);
//            Mod_SystemRx_SetState(STATE_CHIPRX_VIDTIMING_CONFIG);
//        }
//    }
//    
//    if (g_stChipRx.ucRxState > STATE_CHIPRX_VIDEO_CHECK)
//    {
//        if (Mod_MipiRx_VidChk_Stable() == FALSE)
//        {
//            //LTLog(LOG_WARN,"Video Timing Change");
//            g_stChipRx.pHdmiRxNotify(MIPIRX_VIDEO_OFF_EVENT);
//            Mod_SystemRx_SetState(STATE_CHIPRX_VIDTIMING_CONFIG);
//        }
//    }
//}


void Mod_MipiRx_StateHandler(void)
{
    switch (g_stChipRx.ucRxState)
    {
        case STATE_CHIPRX_POWER_ON :
            Mod_MipiRx_Init();
            Mod_SystemRx_SetState(STATE_CHIPRX_WAIT_SOURCE);
        break;
        
        case STATE_CHIPRX_WAIT_SOURCE:
            Drv_MipiRx_PhyPowerOn();
            Drv_MipiRxClk_Sel();
            Drv_System_VidChkClk_SrcSel(MLRX_BYTE_CLK);
            Drv_System_VidChk_SrcSel(MIPIDEBUG);
            Drv_SystemActRx_Sel(MIPIRX);
            Mod_MipiRxDig_Set();
            Mod_SystemRx_SetState(STATE_CHIPRX_VIDTIMING_CONFIG);
        break;
        
        case STATE_CHIPRX_VIDTIMING_CONFIG:
            if(Drv_MipiRx_VidTiming_Get() == true)
            {
                g_stChipRx.ucRxFormat = Drv_MipiRx_VidFmt_Get(g_stMipiRxVidTiming_Get.ucFmt);
                if (Drv_MipiRx_VidTiming_Sel() == true)
                {
                    Mod_SystemRx_SetState(STATE_CHIPRX_PLL_CONFIG);
                }
                else
                {
                    printf("\nNo Video Timing Matched");
                    Mod_SystemRx_SetState(STATE_CHIPRX_WAIT_SOURCE);
                }
            }
        break;
  
        case STATE_CHIPRX_PLL_CONFIG: 
            Drv_MipiRx_DesscPll_Set();
            if(Drv_MipiRx_PcrCali() == SUCCESS)
            {
                printf("\nLT9211C pcr stable");
				Drv_System_VidChkClk_SrcSel(DESSCPLL_PIX_CLK);
                Drv_System_VidChk_SrcSel(MIPIRX);
                Mod_SystemRx_SetState(STATE_CHIPRX_VIDEO_CHECK);
            }
            else
            {
                Mod_SystemRx_SetState(STATE_CHIPRX_VIDTIMING_CONFIG);
            }
        break;

        case STATE_CHIPRX_VIDEO_CHECK: 
            if (Mod_MipiRx_VidChk_Stable() == true)
            {
                printf("\nVideo Check Stable");
                Drv_Csc_Set();
                Drv_MipiRx_VidChkDebug();
                g_stChipRx.pHdmiRxNotify(MIPIRX_VIDEO_ON_EVENT);
                Mod_SystemRx_SetState(STATE_CHIPRX_PLAY_BACK);
            }
        break;

        case STATE_CHIPRX_PLAY_BACK:
        break;
    }
}



void Mod_MipiRx_Handler(void)
{
//    Mod_MipiRx_Statejudge();
    Mod_MipiRx_StateHandler();
    
}

#endif