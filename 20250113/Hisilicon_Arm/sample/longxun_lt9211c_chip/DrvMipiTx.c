#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
  * @project: LT9211C
  * @file: DrvMipiTx.c
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#include "include.h"

#if ((LT9211C_MODE_SEL == LVDS_IN_MIPI_OUT)||(LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)||(LT9211C_MODE_SEL == TTL_IN_MIPI_OUT))


//========================================================================
// Func Name   : Drv_MipiTx_VideoSet
// Description : Mipi Set dsi/csi
// Input       : u8 b1Opt  On/Off
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_VideoSet(u8 b1Opt)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    if (b1Opt == ON)
    {
        #if (MIPITX_OUT_SEL == MIPI_DSI)
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | BIT4_1));
        printf("\nMipi DSI Out");
        #else
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | BIT5_1));
        printf("\nMipi CSI Out");
        #endif
    }
    else
    {
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) & 0xCF));
        
        HDMI_WriteI2C_Byte(0xff,0x81);
        HDMI_WriteI2C_Byte(0x03,0xbf); //mltx reset
    }
    
    
}


//========================================================================
// Func Name   : Drv_MipiTx_GetMipiInClk
// Description : Get Mipi In Clk
// Input       : INOUT StructMipiTx* pstMipiTx  
// Output      : INOUT StructMipiTx* pstMipiTx  
// Return      : u32
//========================================================================
u32 Drv_MipiTx_GetMipiInClk(INOUT StructMipiTx* pstMipiTx)
{
    u32 ulHalfPixClk = 0;
#if LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT
    ulHalfPixClk = Drv_System_FmClkGet(AD_DESSCPLL_PCR_CLK);
#endif

#if (LT9211C_MODE_SEL == LVDS_IN_MIPI_OUT || LT9211C_MODE_SEL == TTL_IN_MIPI_OUT)
    ulHalfPixClk = Drv_System_FmClkGet(AD_DESSCPLL_PIX_CLK) / 2;
#endif    
    
    if ((pstMipiTx->ulMipiInClk != ulHalfPixClk) && (0x00 != ulHalfPixClk))
    {
        pstMipiTx->ulMipiInClk = ulHalfPixClk;
    }
    return ulHalfPixClk;
}

//========================================================================
// Func Name   : Drv_MipiTx_DPhySet
// Description : MIPI Tx DPHY Setting
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_DPhySet(void)
{
    
    
    HDMI_WriteI2C_Byte(0xff,0x82);
    #if (MIPITX_PORT_SEL == PORTA)
    HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) | 0x01)); //0x00: disabled, 0x01: portA en
                                                                //0x02: portB en, 0x03: portA & B en
    printf("\nMIPI Output PortA");
    #elif (MIPITX_PORT_SEL == PORTB)
    HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) | 0x03));
    HDMI_WriteI2C_Byte(0x3c,0x00); //portA HSTX disable
    printf("\nMIPI Output PortB");
    #elif (MIPITX_PORT_SEL == DOU_PORT)
    HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) | 0x03));
    printf("\nMIPI Output PortA & B");
    #endif
    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x53,0xee);

    
    //mipi tx phy cts test require
    //below setting are used for mipitx D-phy cts 1.3.10 when datarate 2.5gbps
    if ((g_stMipiTx.ulMipiDataRate > 1500000) || (g_stMipiTx.ulMipiDataRate <= 1050000)) //mipitx D-Dphy cts 1.4.17 test fail when datarate less than 1.5Gbps if use those setting
    {
        HDMI_WriteI2C_Byte(0xff,0x82);
        HDMI_WriteI2C_Byte(0x37,0x11);
        HDMI_WriteI2C_Byte(0x39,0x32);
        HDMI_WriteI2C_Byte(0x3a,0xc6);
        HDMI_WriteI2C_Byte(0x3b,0x21);

        HDMI_WriteI2C_Byte(0x46,0x4c);
        HDMI_WriteI2C_Byte(0x47,0x4c);
        HDMI_WriteI2C_Byte(0x48,0x48);
        HDMI_WriteI2C_Byte(0x49,0x4c);
        HDMI_WriteI2C_Byte(0x4a,0x4c);
        HDMI_WriteI2C_Byte(0x4b,0x4c);
        HDMI_WriteI2C_Byte(0x4c,0x4c);
        HDMI_WriteI2C_Byte(0x4d,0x48);
        HDMI_WriteI2C_Byte(0x4e,0x4c);
        HDMI_WriteI2C_Byte(0x4f,0x4c);
    }
    
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x03,0xbf); //mltx reset
    HDMI_WriteI2C_Byte(0x03,0xff); //mltx release
}

//========================================================================
// Func Name   : Drv_MipiTx_PllSet
// Description : pll calc loop divider according calc_datarate
// Input       : IN StructMipiTx* pstMipiTx  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_PllSet(IN StructMipiTx* pstMipiTx)
{
    u32 ulMpiTXPhyClk;
    u8 ucSericlkDiv , ucPreDiv, ucDivSet;

    //MIPI D-PHY datarate use full speed <= 1.5Gbps for decrease clk jitter, to solve mipi tx phy cts_1.4.18.
    //use half speed when datarate > 1.5Gbps
    if (g_stMipiTx.ulMipiDataRate <= 1500000)
    {
        ulMpiTXPhyClk = pstMipiTx->ulMipiDataRate;
        HDMI_WriteI2C_Byte(0xff,0x82);
        HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) & BIT4_0));
    }
    else
    {
        ulMpiTXPhyClk = pstMipiTx->ulMipiDataRate / 2;
    }
    
    //txpll sericlk div default use 0x00(DIV2) because mipi tx phy use half rate mode
    HDMI_WriteI2C_Byte(0xff,0x82);

    if (ulMpiTXPhyClk >= 640000 )//640M~1.28G
    {
        ucSericlkDiv = 1; //sericlk div1 [6:4]:0x40
        HDMI_WriteI2C_Byte(0x32,(HDMI_ReadI2C_Byte(0x32) & 0x8f) | 0x40);
    }
    else if (ulMpiTXPhyClk >= 320000 && ulMpiTXPhyClk < 640000)
    {
        ucSericlkDiv = 2; //sericlk div2 [6:4]:0x00
        HDMI_WriteI2C_Byte(0x32,(HDMI_ReadI2C_Byte(0x32) & 0x8f));
    }
    else if (ulMpiTXPhyClk >= 160000 && ulMpiTXPhyClk < 320000)
    {
        ucSericlkDiv = 4; //sericlk div4 [6:4]:0x10
        HDMI_WriteI2C_Byte(0x32,(HDMI_ReadI2C_Byte(0x32) & 0x8f) | 0x10);
    }
    else if (ulMpiTXPhyClk >= 80000 && ulMpiTXPhyClk < 160000)
    {
        ucSericlkDiv = 8; //sericlk div8 [6:4]:0x20
        HDMI_WriteI2C_Byte(0x32,(HDMI_ReadI2C_Byte(0x32) & 0x8f) | 0x20);
    }
    else //40M~80M
    {
        ucSericlkDiv = 16; //sericlk div16 [6:4]:0x30
        HDMI_WriteI2C_Byte(0x32,(HDMI_ReadI2C_Byte(0x32) & 0x8f) | 0x30);
    }

    printf("\nulMipiDataRate:%ld, ulHalfPixClk:%ld, ulMpiTXPhyClk:%ld, ucBpp:%d, ucTxPortNum:0x%02x, ucTxLaneNum:0x%02x",
                pstMipiTx->ulMipiDataRate,pstMipiTx->ulMipiInClk, ulMpiTXPhyClk, pstMipiTx->ucBpp, pstMipiTx->ucTxPortNum,pstMipiTx->ucTxLaneNum);

    //prediv_set N1 = 1
    ucPreDiv = 1;
    
    //div set
    //Vcoclk=byte_clk*4*M3=25M*M1*M2*ucSericlkDiv(�첽ģʽ), M2 default value is 2;
    ucDivSet = (u8)(ulMpiTXPhyClk * ucSericlkDiv / 25000);

    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x30,0x02); //[7]0:txpll normal work
                                   //[2:1]Lmtxpll reference clock selection:Xtal clock;
    HDMI_WriteI2C_Byte(0x31,0x28);
    HDMI_WriteI2C_Byte(0x32,(HDMI_ReadI2C_Byte(0x32) & 0xf3)); //tx pll post div set DIV1
    HDMI_WriteI2C_Byte(0x34,0x01);
    HDMI_WriteI2C_Byte(0x35,ucDivSet);

    printf("\nucSericlkDiv N1:0x%02x, ucDivSet M2:0x%02x",ucSericlkDiv, ucDivSet);
}
//========================================================================
// Func Name   : Drv_MipiTx_PllCali
// Description : first pll cali, then phy set
// Input       : void  
// Output      : None
// Return      : u8
//========================================================================
u8 Drv_MipiTx_PllCali(void)
{
    u8 ucPllScanCnt = 0;
    u8 ucRtn = false;
    usleep(20*1000);
    do 
    {    
        HDMI_WriteI2C_Byte(0xff,0x81);//register bank	
        HDMI_WriteI2C_Byte(0x0c,0x78);//tx pll rest cal_rst =0
        HDMI_WriteI2C_Byte(0xff,0x87);//register bank
        HDMI_WriteI2C_Byte(0x0f,0x00);//tx pll cal = 0;
        
        HDMI_WriteI2C_Byte(0xff,0x81);//register bank
        HDMI_WriteI2C_Byte(0x0c,0xf9);//tx pll rest cal_rst =1
        HDMI_WriteI2C_Byte(0xff,0x87);//register bank
        HDMI_WriteI2C_Byte(0x0f,0x01);//tx pll cal = 0;
        usleep(10*1000);
        ucPllScanCnt++;
    }while((ucPllScanCnt < 3) && ((HDMI_ReadI2C_Byte(0x39) & 0x07) != 0x05));//PLL calibration done status
    if((HDMI_ReadI2C_Byte(0x39) & 0x07)== 0x05)
    {
        ucRtn = SUCCESS;
        printf("\nTx Pll Lock");
    }
    else
    {
        ucRtn = FAIL;
        printf("\nTx Pll Unlocked");
    }
    return ucRtn;
}

//========================================================================
// Func Name   : Drv_MipiTx_PortDataEnable
// Description : RGD_MPTX_PT0/PT1_TXN_CMD_EN
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_PortDataEnable(void)
{
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x4b,(HDMI_ReadI2C_Byte(0x4b) | 0x02)); //portA lane0
    HDMI_WriteI2C_Byte(0x4c,(HDMI_ReadI2C_Byte(0x4c) | 0x01)); //portA lane1
    HDMI_WriteI2C_Byte(0x4d,(HDMI_ReadI2C_Byte(0x4d) | 0x01)); //portA lane2
    HDMI_WriteI2C_Byte(0x4e,(HDMI_ReadI2C_Byte(0x4e) | 0x01)); //portA lane3
    HDMI_WriteI2C_Byte(0x4f,(HDMI_ReadI2C_Byte(0x4f) | 0x01)); //portA lane4

    HDMI_WriteI2C_Byte(0x51,(HDMI_ReadI2C_Byte(0x51) | 0x02)); //portB lane0
    HDMI_WriteI2C_Byte(0x52,(HDMI_ReadI2C_Byte(0x52) | 0x01)); //portB lane1
    HDMI_WriteI2C_Byte(0x53,(HDMI_ReadI2C_Byte(0x53) | 0x01)); //portB lane2
    HDMI_WriteI2C_Byte(0x54,(HDMI_ReadI2C_Byte(0x54) | 0x01)); //portB lane3
    HDMI_WriteI2C_Byte(0x55,(HDMI_ReadI2C_Byte(0x55) | 0x01)); //portB lane4
}

//========================================================================
// Func Name   : Drv_MipiTx_PhyTimingParaSet
// Description : mipi phy timing set
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_PhyTimingParaSet(INOUT StructMipiTx* pstMipiTx, INOUT StructMipiTxDPhy* pstMipiTxDPhy)
{

    u32 ulrdbyteclk  = 0;
    
    ulrdbyteclk = Drv_System_FmClkGet(AD_MLTX_WRITE_CLK);
    ulrdbyteclk = ulrdbyteclk / 1000;
    printf("\nbyteclk: %ldM", ulrdbyteclk);

    g_stMipiTxDPhy.ucClkPre = 0x02;
    pstMipiTxDPhy->ucHsLpx   = ulrdbyteclk * 6 / 100 + 1; //hs lpx > 50ns
    pstMipiTxDPhy->ucHsPrep  = ulrdbyteclk * 6 / 100; //hs prep : (40ns + 4*UI)~(85ns + 6*UI) , clk_prepare
    pstMipiTxDPhy->ucHsTrail = ulrdbyteclk * 7 / 100 + 4; //hs_trail and clk_trail: max(8UI , 60ns + 4UI),   +2->+3
    pstMipiTxDPhy->ucClkPost = ulrdbyteclk * 7 / 100 + 7;//ck_post > 60ns + 52UI,  +4->+15->+1
        
    if(pstMipiTx->b1MipiClockburst)
    {

        //������ʱ�Ӵ�1.94G~2.5G�£�clk_zero���ݹ̶���ʽ�޷���ͼ
        if (pstMipiTx->ulMipiDataRate > CTS_DATARATE)
        {
            pstMipiTxDPhy->ucClkZero = 0x05;
        }
        else
        {
            pstMipiTxDPhy->ucClkZero = ulrdbyteclk * 6 / 25;    //ck_zero > 300 - ck_prpr , old: rdbyteclk/4      
        }
        
        pstMipiTxDPhy->ucHsRqStPre = g_stMipiTxDPhy.ucHsLpx + g_stMipiTxDPhy.ucHsPrep + g_stMipiTxDPhy.ucClkZero + g_stMipiTxDPhy.ucClkPre;
    }
    else
    {
        pstMipiTxDPhy->ucClkZero   = ulrdbyteclk * 6 / 25;    //ck_zero > 300 - ck_prpr , old: rdbyteclk/4      
        pstMipiTxDPhy->ucHsRqStPre = ulrdbyteclk / 10;
    }
    
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x8a,pstMipiTxDPhy->ucHsRqStPre);
    HDMI_WriteI2C_Byte(0xa4,pstMipiTxDPhy->ucHsLpx);
    HDMI_WriteI2C_Byte(0xa5,pstMipiTxDPhy->ucHsPrep);
    HDMI_WriteI2C_Byte(0xa6,pstMipiTxDPhy->ucHsTrail);
    HDMI_WriteI2C_Byte(0xa7,pstMipiTxDPhy->ucClkZero);
    HDMI_WriteI2C_Byte(0xa9,pstMipiTxDPhy->ucClkPost);

    printf("\nck_post (0xD4A9) = 0x%02x", pstMipiTxDPhy->ucClkPost);
    printf("\nck_zero (0xD4A7) = 0x%02x", pstMipiTxDPhy->ucClkZero);
    printf("\nhs_lpx  (0xD4A4) = 0x%02x", pstMipiTxDPhy->ucHsLpx);
    printf("\nhs_prep (0xD4A5) = 0x%02x", pstMipiTxDPhy->ucHsPrep);
    printf("\nhs_trail(0xD4A6) = 0x%02x", pstMipiTxDPhy->ucHsTrail);
    printf("\nhs_rqst (0xD48A) = 0x%02x", pstMipiTxDPhy->ucHsRqStPre);
}

//========================================================================
// Func Name   : Drv_MipiTx_TimingSet
// Description : mipi tx timing set
// Input       : IN StructVidChkTiming *pstVidTiming  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_TimingSet(IN StructVidChkTiming *pstVidTiming)
{
    u16 ushss, usvss;
    u16 us3d_dly;
 
    ushss = pstVidTiming->usHs + pstVidTiming->usHbp;
    usvss = pstVidTiming->usVs + pstVidTiming->usVbp;
    
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x7e,(u8)(pstVidTiming->usHact >> 8));
    HDMI_WriteI2C_Byte(0x7f,(u8)pstVidTiming->usHact);
    HDMI_WriteI2C_Byte(0x7c,(u8)(pstVidTiming->usVact >> 8));
    HDMI_WriteI2C_Byte(0x7d,(u8)pstVidTiming->usVact);
    HDMI_WriteI2C_Byte(0x84,(u8)(HDMI_ReadI2C_Byte(0x84) | ((ushss >> 8) & 0xfc)));
    HDMI_WriteI2C_Byte(0x85,(u8)ushss);
    HDMI_WriteI2C_Byte(0x80,(u8)(usvss >> 8));
    HDMI_WriteI2C_Byte(0x81,(u8)usvss);

    //3d dly
    us3d_dly = pstVidTiming->usHact / 4;
    HDMI_WriteI2C_Byte(0x7a,(u8)(us3d_dly >> 8));
    HDMI_WriteI2C_Byte(0x7b,(u8)us3d_dly);
}

//========================================================================
// Func Name   : Drv_MipiTx_InHSyncPol_Sel
// Description : Rx     hsync/vsync 1:positive 0:negative
//               MIPITX hsync/vsync 1:negative 0:positive
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_InHSyncPol_Sel(IN u8 b1SyncPol)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    if (b1SyncPol == NEGITVE)
    {
        HDMI_WriteI2C_Byte(0x70,(HDMI_ReadI2C_Byte(0x70) | BIT1_1));
    }
    else
    {
        HDMI_WriteI2C_Byte(0x70,(HDMI_ReadI2C_Byte(0x70) & BIT1_0));
    }
}

//========================================================================
// Func Name   : Drv_MipiTx_InVSyncPol_Sel
// Description : Rx     hsync/vsync 1:positive 0:negative
//               MIPITX hsync/vsync 1:negative 0:positive
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_InVSyncPol_Sel(IN u8 b1SyncPol)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    if (b1SyncPol == NEGITVE)
    {
        HDMI_WriteI2C_Byte(0x70,(HDMI_ReadI2C_Byte(0x70) | BIT0_1));
    }
    else
    {
        HDMI_WriteI2C_Byte(0x70,(HDMI_ReadI2C_Byte(0x70) & BIT0_0));
    }
}

//========================================================================
// Func Name   : Drv_MipiTx_Hss_Set
// Description : Hss Set
// Input       : IN u16 value  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_Hss_Set(IN u16 usVal)
{
    if( usVal > 0x3FF )
    {
        usVal = 0x3FF;
    }

    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x84,(u8)(HDMI_ReadI2C_Byte(0x84) | (usVal >> 8)));
    HDMI_WriteI2C_Byte(0x85,(u8)usVal);
}

//========================================================================
// Func Name   : Drv_MipiTx_Hss_Get
// Description : Hss Get
// Input       : void  
// Output      : None
// Return      : u16
//========================================================================
u16 Drv_MipiTx_Hss_Get(void)
{
    u16 rdhss = 0;

    HDMI_WriteI2C_Byte(0xff,0xd4);
    rdhss = (HDMI_ReadI2C_Byte(0x84) & 0x03); 
    rdhss = (rdhss << 8); 
    rdhss = rdhss + (HDMI_ReadI2C_Byte(0x85));
    return rdhss;
}

//========================================================================
// Func Name   : Drv_MipiTx_FSMHact_Get
// Description : Get Fsm Hact
// Input       : void  
// Output      : None
// Return      : u16
//========================================================================
u16 Drv_MipiTx_FSMHact_Get(void)
{
    u16 rgod_hact = 0;
    HDMI_WriteI2C_Byte(0xff,0xd4);
    rgod_hact = (HDMI_ReadI2C_Byte(0xc1) & 0x3f);
    rgod_hact = (rgod_hact << 8);
    rgod_hact = rgod_hact + HDMI_ReadI2C_Byte(0xc2);
    return rgod_hact;
}

//========================================================================
// Func Name   : Drv_MipiTx_FifoDelay_Set
// Description : MIPI Tx rddly set
// Input       : IN u16 rddly  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_FifoDelay_Set(IN u16 rddly)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x82,(u8)(rddly >> 8));
    HDMI_WriteI2C_Byte(0x83,rddly);
    

}


void Drv_MipiTx_DPHYCSI8Lane_En(IN u8 b1IsEn)
{
    HDMI_WriteI2C_Byte(0Xff,0xd4);

    if (b1IsEn)
    {
        HDMI_WriteI2C_Byte(0Xa0,(HDMI_ReadI2C_Byte(0xa0) | 0x40));
    }
    else
    {
        HDMI_WriteI2C_Byte(0Xa0,(HDMI_ReadI2C_Byte(0xa0) & 0xbf));
    }
}

//========================================================================
// Func Name   : Drv_MipiTx_LaneSet
// Description : mipi tx lane set
//                    MIPI lane mode:
//                    2'b00 = 4lane;
//                    2'b01 = 1lane;
//                    2'b10 = 2lane;
//                    2'b11 = 3lane.
// Input       : u8 LaneNum  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_LaneSet(IN u8 LaneNum)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) & 0xfc));
	if(LaneNum != MIPITX_4LANE)
	{
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | LaneNum));
	}
}

//========================================================================
// Func Name   : Drv_MipiTx_PortCopy
// Description : 
//                Physical port0/1 output data source select:
//                1'b0 = Output logic port 0;
//                1'b1 = Output logic port 1.
// Input       : IN u8 ucTxPortNum  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_PortCopy()
{

    #if ((MIPITX_PORT_SEL == DOU_PORT) || (MIPITX_PORT_SEL == PORTA))
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x4a,(HDMI_ReadI2C_Byte(0x4a) | 0x40));
    HDMI_WriteI2C_Byte(0x50,(HDMI_ReadI2C_Byte(0x50) | 0x40));
    #endif
}

void Drv_MipiTx_PortSet(IN u8 ucTxPortNum)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x87,(HDMI_ReadI2C_Byte(0x87) | ucTxPortNum)); //single port
}



//========================================================================
// Func Name   : Drv_MipiTx_DPHYClkMode_Sel
// Description : CLOCK LANE non-burst enable.
// Input       : IN u8 b1MipiClockburst  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_DPHYClkMode_Sel(IN u8 b1IsMipiClockburst)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    if (b1IsMipiClockburst)
    {
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | 0x80));
    }
    else
    {
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) & 0x7f));
    }
}

//========================================================================
// Func Name   : Drv_MipiTx_DPHYClkData_Set
// Description : default value is 0x55, need set 0xAA
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_DPHYClkData_Set(void)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0xaa,0xAA);
}


//========================================================================
// Func Name   : Drv_MipiTx_HalfWrClkSrc_Sel
// Description : MPTX write clk select:
//                0 = Select half write clk;
//                1 = Select write clk.
// Input       : IN u8 clock  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_HalfWrClkSrc_Sel(IN u8 clock)
{
    HDMI_WriteI2C_Byte(0xff,0x82);
    
    if (clock == HALF_WRITE_CLK)
    {
        HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) | 0x80));
    }
    else
    {
        HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) & 0x7f));
    }
}

//========================================================================
// Func Name   : Drv_MipiTx_DsiDataTypeSet
// Description : DSI DataType Set
// Input       : IN u8 ucTxFormat  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_DsiDataTypeSet(IN u8 ucTxFormat)
{
    //Data Type set
    HDMI_WriteI2C_Byte(0xff,0xd4);
    switch (ucTxFormat)
    {
        case RGB_6Bit: //bpp18
            HDMI_WriteI2C_Byte(0x88,0x10); //rgb666
            HDMI_WriteI2C_Byte(0x86,0x1e);       
            break;
        case RGB_6Bit_L: //bpp24
            HDMI_WriteI2C_Byte(0x88,0x22); //rgb666_loosely
            HDMI_WriteI2C_Byte(0x86,0x2e);
            break;
        case RGB_565Bit: //bpp16
            HDMI_WriteI2C_Byte(0x88,0x00); //rgb565
            HDMI_WriteI2C_Byte(0x86,0x0e);
            break;
        case RGB_8Bit: //bpp24
            HDMI_WriteI2C_Byte(0x88,0x21);
            HDMI_WriteI2C_Byte(0x86,0x3e);
            break;
        case RGB_10Bit: //bpp30
            HDMI_WriteI2C_Byte(0x88,0x41);
            HDMI_WriteI2C_Byte(0x86,0x0d); //rgb30
            break;
        case YUV422_8bit: //bpp16
            HDMI_WriteI2C_Byte(0x88,0x01);
            HDMI_WriteI2C_Byte(0x86,0x2c); //yuv16
            break;
        case YUV422_10bit: //bpp20
            HDMI_WriteI2C_Byte(0x88,0x24);
            HDMI_WriteI2C_Byte(0x86,0x0c); //yuv_20
            break;
        case YUV422_12bit: //bpp24
            HDMI_WriteI2C_Byte(0x88,0x23);
            HDMI_WriteI2C_Byte(0x86,0x1c); //yuv24
            break;
        case YUV420_8bit: //bpp24
            HDMI_WriteI2C_Byte(0x88,0x20);
            HDMI_WriteI2C_Byte(0x86,0x3d); //yuv420
            break;
        case YUV420_10bit: //bpp30
            break;
       
        default:
            HDMI_WriteI2C_Byte(0x88,0x21);
            HDMI_WriteI2C_Byte(0x86,0x3e);
            break;
    }
}

//========================================================================
// Func Name   : Drv_MipiTx_CsiDataTypeSet
// Description : CSI DataType Set
// Input       : IN u8 ucTxFormat  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_CsiDataTypeSet(IN u8 ucTxFormat)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    switch (ucTxFormat)
    {
        case RGB_6Bit : //bpp18
            HDMI_WriteI2C_Byte(0x88,0x11);
            HDMI_WriteI2C_Byte(0x86,0x23); //csi_rgb666         
            break;
        case RGB_565Bit : //bpp16
            HDMI_WriteI2C_Byte(0x88,0x02);
            HDMI_WriteI2C_Byte(0x86,0x22); //csi_rgb565         
            break;
        case RGB_8Bit : //bpp24
            HDMI_WriteI2C_Byte(0x88,0x25);
            HDMI_WriteI2C_Byte(0x86,0x24); //csi_rgb888         
            break;
        case RGB_10Bit : //bpp30
            break;
        case YUV422_8bit ://bpp16
            HDMI_WriteI2C_Byte(0x88,0x01);
            HDMI_WriteI2C_Byte(0x86,0x1e); //yuv16
            break;
        case YUV422_10bit ://bpp20
            HDMI_WriteI2C_Byte(0x88,0x31);
            HDMI_WriteI2C_Byte(0x86,0x1f); //Y422_1_10
            break;
        case YUV420_8bit :
            HDMI_WriteI2C_Byte(0x88,0x20);
            HDMI_WriteI2C_Byte(0x86,0x1a); //Y420_3_legacy
            break;
        case YUV420_10bit :
            HDMI_WriteI2C_Byte(0x88,0x41);
            HDMI_WriteI2C_Byte(0x86,0x19); //Y420_3_10bit
            break;
    }
}

//========================================================================
// Func Name   : Drv_MipiTx_CsiFrameCntSet
// Description : CSI FrameCnt Set
// Input       : Void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_CsiFrameCntSet(void)
{
    #if LT9211C_VERSION == U1 
    //P_FORMAT: 1212
    //I_FORMAT: 1212  (not distinguish Odd-even field)
    #elif LT9211C_VERSION == U2 
    //P_FORMAT: 0000
    //I_FORMAT: 1212  (distinguish Odd-even field)
        #if MIPITX_VIDEO_FORMAT == P_FORMAT
            HDMI_WriteI2C_Byte(0xff,0xd4);
            HDMI_WriteI2C_Byte(0xb9,0x10);
        #endif
    #endif
}

//========================================================================
// Func Name   : Drv_MipiTx_DcsAutoMode
// Description : 111
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_DcsAutoMode(void)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) | BIT6_1));
    HDMI_WriteI2C_Byte(0xb6,0x10);
}

//========================================================================
// Func Name   : Drv_MipiTx_DPhySkewCali
// Description : DPHY Skew Cali
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_DPhySkewCali(void)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);

    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) | 0x0F)); //RGD_SKEW_CALI_PT_CFG[3:0]
    HDMI_WriteI2C_Byte(0xac,0x18); //RGD_SKEW_CALI_LEN[15:8]
    HDMI_WriteI2C_Byte(0xad,0x00); //RGD_SKEW_CALI_LEN[7:0]
    HDMI_WriteI2C_Byte(0xae,0x20); //RGD_SKEW_CALI_HS_ZERO[7:0]
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) | BIT4_1));
    usleep(10*1000);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) & BIT4_0));
    HDMI_WriteI2C_Byte(0xae,0x00);
}

//========================================================================
// Func Name   : Drv_MipiTx_DPhyClkHsTrig
// Description : DPHY HS Clk Trig
// Input       : void  
// Output      : None
// Return      : void
//========================================================================
void Drv_MipiTx_DPhyClkHsTrig(void)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) & 0xdf));
    usleep(5*1000);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) | BIT5_1)); //[5]:Dphy clk lane hs mode initial trigger
    usleep(1*1000);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) & 0xdf));
}


#endif
