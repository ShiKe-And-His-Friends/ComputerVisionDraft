/******************************************************************************
*  Copyright (C), 2006-2022, Lontium Tech.
*  Project       : LT2103
*  File Name     : ModMipiTx.h
*  Version       : V1.0
*  Author        : sxue
*  Created       : 2022/7/15
*  Description   : 
*  
*  History:
*  2022/7/15     sxue      Created File
******************************************************************************/

#ifndef _MODMIPITX_H
#define _MODMIPITX_H


#if ((LT9211C_MODE_SEL == LVDS_IN_MIPI_OUT)||(LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)||(LT9211C_MODE_SEL == TTL_IN_MIPI_OUT))


#define     MIPITX_OUT_SEL             MIPI_DSI            //MIPI_DSI/MIPI_CSI
#define     MIPITX_PORT_SEL            DOU_PORT               //PORTA/PORTB/DOU_PORT
#define     MIPITX_LANE_NUM            MIPITX_4LANE        //MIPITX_4LANE/MIPITX_3LANE/MIPITX_2LANE/MIPITX_1LANE, no MIPITX_8LANE
#define     MIPI_CLOCK_BURST           DISABLED            //ENABLED/DISABLED
#define     MIPITX_VIDEO_FORMAT        P_FORMAT            //P_FORMAT/I_FORMAT 
#define     CSC_TX_MODE                RGB                 //RGB/YUV422
#define     MIPI_TX_FORMAT             RGB_8Bit          //RGB_6Bit, RGB_8Bit,YUV422_8bit,RGB_6Bit_L,RGB_565Bit

   










extern void Mod_MipiTx_ParaSet(void);
extern void Mod_MipiTx_Handler(void);

#endif

#endif