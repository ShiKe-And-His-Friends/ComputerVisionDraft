/******************************************************************************
  * @project: LT9211C
  * @file: ModMipiRpt.c
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#include    "include.h"


#if (LT9211C_MODE_SEL == MIPI_REPEATER)


void Mod_MipiRptClkStb_Judge()
{
    if (g_stChipRx.ucRxState > STATE_CHIPRX_VIDEO_CHECK)
    {
        if(Drv_MipiRptClk_Change() == TRUE)
        {
            printf("MIPI Repeater Rx Clk Change");
            Mod_SystemRx_SetState(STATE_CHIPRX_WAIT_SOURCE);
        }
    }    
}


void Mod_MipiRpt_StateHandler()
{
    switch (g_stChipRx.ucRxState)
    {
        case STATE_CHIPRX_POWER_ON:
            Mod_SystemRx_SetState(STATE_CHIPRX_WAIT_SOURCE);
        break;    
        
        case STATE_CHIPRX_WAIT_SOURCE:
            Drv_MipiRpt_ClkSel();
            Drv_MipiRptTx_SkewCali();
            Drv_MipiRptRxPhy_Set();
            Drv_MipiRptRxDig_Set();
            Mod_SystemRx_SetState(STATE_CHIPRX_PLL_CONFIG);
        break;
        
        case STATE_CHIPRX_PLL_CONFIG:
            if (Drv_MipiRptTx_PllSet() == TRUE)
            {
                if (Drv_MipiRptTx_PllCali() == TRUE)
                {
                    Mod_SystemRx_SetState(STATE_CHIPRX_VIDEO_CHECK);
                }
                else
                {
                    Mod_SystemRx_SetState(STATE_CHIPRX_VIDEO_CHECK);//STATE_CHIPRX_WAIT_SOURCE
                }
            }
        break;
            
        case STATE_CHIPRX_VIDEO_CHECK:
            Drv_MipiRptTxPhy_Set();
            Drv_MipiRptBta_Set();
            Mod_SystemRx_SetState(STATE_CHIPRX_PLAY_BACK);
        break;
        
        case STATE_CHIPRX_PLAY_BACK:
        break;
    }
}

void Mod_MipiRpt_Handler()
{
//    Mod_MipiRptClkStb_Judge();
    Mod_MipiRpt_StateHandler();
}

#endif

