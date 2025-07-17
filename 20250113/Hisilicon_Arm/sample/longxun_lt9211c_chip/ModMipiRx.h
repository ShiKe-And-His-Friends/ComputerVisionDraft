/******************************************************************************
  * @project: LT9211C
  * @file: ModMipiRx.h
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#ifndef _MODMIPIRX_H
#define _MODMIPIRX_H

#if ((LT9211C_MODE_SEL == MIPI_IN_LVDS_OUT)||(LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)||(LT9211C_MODE_SEL == MIPI_IN_TTL_OUT))



#define     MIPIRX_INPUT_SEL           MIPI_DSI            //MIPI_DSI/MIPI_CSI
#define     MIPIRX_PORT_SEL            PORTA              //PORTA/PORTB
#define     MIPIRX_LANE_NUM            MIPIRX_4LANE        //MIPIRX_4LANE/MIPIRX_3LANE/MIPIRX_2LANE/MIPIRX_1LANE
#define     MIPIRX_CLK_BURST           DISABLED             //ENABLED/DISABLED
#define     CSC_RX_MODE                RGB             //RGB/YUV422




extern u8 Mod_MipiRx_VidChk_Stable();
extern void Mod_MipiRx_Statejudge(void);
extern void Mod_MipiRx_Handler(void);

#endif

#endif