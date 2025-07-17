#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
  * @project: LT9211C
  * @file: DrvMipiRx.c
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#include "include.h"

#if ((LT9211C_MODE_SEL == MIPI_IN_LVDS_OUT)||(LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)||(LT9211C_MODE_SEL == MIPI_IN_TTL_OUT))

StructPcrPara g_stPcrPara;
SrtuctMipiRx_VidTiming_Get g_stMipiRxVidTiming_Get;

//hfp    hs     hbp     hact     htotal   vfp   vs   vbp   vact    vtotal  framerate
static struct VideoTimingList resolution[]  = {
 {24,    96,    40,     640,     800,    33,    2,   10,   480,    525,   60},  //video_640x480_60Hz
 {16,    62,    60,     720,     858,     9,    6,   30,   480,    525,   60},  //video_720x480_60Hz
 {12,    64,    88,     720,     864,     5,    5,   39,   576,    625,   50},  //video_720x576_50Hz
// {80,    20,   80,     800,     980,    20,    4,   12,   1280,    1316,   60},  //video_800x600_60Hz
  {16,    16,   48,     800,     880,    8,    8,   16,   1280,    1312,   60},  //video_800x600_60Hz
 {110,   40,    220,    1280,    1650,    5,    5,   20,   720,    750,   30},  //video_1280x720_30Hz
 {440,   40,    220,    1280,    1980,    5,    5,   20,   720,    750,   50},  //video_1280x720_50Hz
 {110,   40,    220,    1280,    1650,    5,    5,   20,   720,    750,   60},  //video_1280x720_60Hz
 {24,    136,   160,    1024,    1344,    3,    6,   29,   768,    806,   60},  //video_1024x768_60Hz
 {26,    110,   110,    1366,    1592,    13,   6,   13,   768,    800,   60},  //video_1366x768_60Hz
 {110,   40,    220,    1280,    1650,    5,    5,   20,   720,    750,   30},  //video_1280x720_30Hz
 {48,    32,    80,     1920,    2080,    5,    5,   20,   720,    750,   60},  //video_1920x720_60Hz
 {48,    112,   248,    1280,    1688,    1,    3,   38,   1024,   1066,  60},  //video_1028x1024_60Hz
 {88,    44,    148,    1920,    2200,    4,    5,   36,   1080,   1125,  30},  //video_1920x1080_30Hz
 {88,    44,    148,    1920,    2200,    4,    5,   36,   1080,   1125,  60},  //video_1920x1080_60Hz
 {88,    44,    148,    1920,    2200,    4,    5,   36,   1080,   1125,  90},  //video_1920x1080_90Hz
// {90,    44,    148,    1920,    2202,    14,    5,   36,   1080,   1135,  60},  
 {64,    192,   304,    1600,    2160,    1,    3,   46,   1200,   1250,  60},  //video_1600x1200_60Hz
 {48,    32,    80,     1920,    2080,    3,    6,   26,   1200,   1235,  60},  //video_1920x1200_60Hz
 {32,    48,    80,     2048,    2208,    6,    3,   28,   1280,   1317,  60},  //video_2048x1280_60Hz
 {50,    48,    80,     2304,    2482,    6,    3,   32,   1440,   1481,  60},  //video_2304x1440_60Hz
 {48,    32,    80,     2560,    2720,    3,    5,   33,   1440,   1481,  60},  //video_2560x1440_60Hz
 {1276,  88,    296,    3840,    5500,    8,    10,  72,   2160,   2250,  24},  //video_3840x2160_24Hz
 {1056,  88,    296,    3840,    5280,    8,    10,  72,   2160,   2250,  25},  //video_3840x2160_25Hz
 {176,   88,    296,    3840,    4400,    8,    10,  72,   2160,   2250,  30},  //video_3840x2160_30Hz

 };
 
#define MIPIRX_FORMAT_CNT   0x0f
static char* g_szStrRxFormat[MIPIRX_FORMAT_CNT] = 
{
    "",
    "DSI YUV422 10bit",
    "DSI YUV422 12bit",
    "YUV422 8bit",
    "RGB 10bit",
    "RGB 12Bit",
    "YUV420 8bit",
    "RGB 565",
    "RGB 666",
    "DSI RGB 6L",
    "RGB 8Bit",
    "RAW8",
    "RAW10",
    "RAW12",
    "CSI YUV422 10",
};

void DRV_DesscPll_SdmCal(void)
{
    HDMI_WriteI2C_Byte(0xff,0xd0);// 	 
    HDMI_WriteI2C_Byte(0x08,0x00);//sel mipi rx sdm  

    HDMI_WriteI2C_Byte(0x26,0x80 | ((u8)g_stPcrPara.Pcr_M)); //m
    HDMI_WriteI2C_Byte(0x2d,g_stPcrPara.Pcr_UpLimit); //PCR M overflow limit setting.
    HDMI_WriteI2C_Byte(0x31,g_stPcrPara.Pcr_DownLimit); //PCR M underflow limit setting. 
    
    HDMI_WriteI2C_Byte(0x27,(u8)(g_stPcrPara.Pcr_K >> 16)); //k
    HDMI_WriteI2C_Byte(0x28,(u8)(g_stPcrPara.Pcr_K >> 8)); //k
    HDMI_WriteI2C_Byte(0x29,(u8)(g_stPcrPara.Pcr_K)); //k  
    HDMI_WriteI2C_Byte(0x26,(HDMI_ReadI2C_Byte(0x26) & 0x7f));
   
}

void Drv_MipiRx_DesscPll_Set()
{
    u8 ucdesscpll_pixck_div = 0;

    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x26,0x20); //[7:6]desscpll reference select Xtal clock as reference
                                   //[4]1'b0: dessc-pll power down
    HDMI_WriteI2C_Byte(0x27,0x40); //prediv = 0;

    printf("\nMipi Rx PixClk: %ld",g_stRxVidTiming.ulPclk_Khz);
    if (g_stRxVidTiming.ulPclk_Khz >= 352000)
    {
        HDMI_WriteI2C_Byte(0x2f,0x04);
        ucdesscpll_pixck_div = 2;
    }
    else if (g_stRxVidTiming.ulPclk_Khz >= 176000 && g_stRxVidTiming.ulPclk_Khz < 352000)
    {
        HDMI_WriteI2C_Byte(0x2f,0x04);
        ucdesscpll_pixck_div = 2;
    }
    else if (g_stRxVidTiming.ulPclk_Khz >= 88000 && g_stRxVidTiming.ulPclk_Khz < 176000)
    {
        HDMI_WriteI2C_Byte(0x2f,0x05);
        ucdesscpll_pixck_div = 4;
    }
    else if (g_stRxVidTiming.ulPclk_Khz >= 44000 && g_stRxVidTiming.ulPclk_Khz < 88000)
    {
        HDMI_WriteI2C_Byte(0x2f,0x06);
        ucdesscpll_pixck_div = 8;
    }
    else if (g_stRxVidTiming.ulPclk_Khz >= 22000 && g_stRxVidTiming.ulPclk_Khz < 44000)
    {
        HDMI_WriteI2C_Byte(0x2f,0x07);
        ucdesscpll_pixck_div = 16;
    }
    else 
    {
        HDMI_WriteI2C_Byte(0x2f,0x07);
        ucdesscpll_pixck_div = 16;
        HDMI_WriteI2C_Byte(0x2c,0x01); //desscpll lowf pixck divsel: /2
    }

    g_stPcrPara.Pcr_M = (g_stRxVidTiming.ulPclk_Khz * ucdesscpll_pixck_div) / 25;
    g_stPcrPara.Pcr_K = g_stPcrPara.Pcr_M % 1000;
    g_stPcrPara.Pcr_M = g_stPcrPara.Pcr_M / 1000;
    
    g_stPcrPara.Pcr_UpLimit   = g_stPcrPara.Pcr_M + 1;
    g_stPcrPara.Pcr_DownLimit = g_stPcrPara.Pcr_M - 1;

    g_stPcrPara.Pcr_K <<= 14;

    DRV_DesscPll_SdmCal();
    HDMI_WriteI2C_Byte(0xff,0x81);	  
    HDMI_WriteI2C_Byte(0x03,0xfe); //desscpll rst
    usleep(1*1000);
    HDMI_WriteI2C_Byte(0x03,0xff); //desscpll rst   
}

u8 Drv_MipiRx_PcrCali(void)
{    
    u8 ucRtn = true;
    u8 ucPcr_Cal_Cnt;

    HDMI_WriteI2C_Byte(0xff,0xd0);
    #if MIPIRX_INPUT_SEL == MIPI_DSI 
    HDMI_WriteI2C_Byte(0x0a,(HDMI_ReadI2C_Byte(0x0a) & 0x0f));
    HDMI_WriteI2C_Byte(0x1e,0x51); //0x51[7:4]pcr diff first step,[3:0]pcr diff second step
    HDMI_WriteI2C_Byte(0x23,0x80); //0x05 MIPIRX PCR capital data set for PCR second step
    HDMI_WriteI2C_Byte(0x24,0x70);
    HDMI_WriteI2C_Byte(0x25,0x80);
    HDMI_WriteI2C_Byte(0x2a,0x10); 
    HDMI_WriteI2C_Byte(0x21,0x4f);
    HDMI_WriteI2C_Byte(0x22,0xf0);
    
    HDMI_WriteI2C_Byte(0x38,0x04); //MIPIRX PCR de mode delay offset0
    HDMI_WriteI2C_Byte(0x39,0x08);
    HDMI_WriteI2C_Byte(0x3a,0x10);
    HDMI_WriteI2C_Byte(0x3b,0x20); //MIPIRX PCR de mode delay offset2

    HDMI_WriteI2C_Byte(0x3f,0x04); //PCR de mode step0 setting
    HDMI_WriteI2C_Byte(0x40,0x08);
    HDMI_WriteI2C_Byte(0x41,0x10);
    HDMI_WriteI2C_Byte(0x42,0x20);
    HDMI_WriteI2C_Byte(0x2b,0xA0);
//    HDMI_WriteI2C_Byte(0x4d,0x00); //0x00
    #elif MIPIRX_INPUT_SEL == MIPI_CSI
    HDMI_WriteI2C_Byte(0x0a,(HDMI_ReadI2C_Byte(0x0a) & 0x1f));
    HDMI_WriteI2C_Byte(0x1e,0x11); //0x51[7:4]pcr diff first step,[3:0]pcr diff second step
    HDMI_WriteI2C_Byte(0x23,0x60); //0x05 MIPIRX PCR capital data set for PCR second step
    HDMI_WriteI2C_Byte(0x24,0x30);
    HDMI_WriteI2C_Byte(0x25,0x80);
    HDMI_WriteI2C_Byte(0x2a,0x10); 
    HDMI_WriteI2C_Byte(0x21,0xc6); //[7]1'b1:CSI sel de sync mode;
    HDMI_WriteI2C_Byte(0x22,0x00);
    
    HDMI_WriteI2C_Byte(0x38,0x04); //MIPIRX PCR de mode delay offset0
    HDMI_WriteI2C_Byte(0x39,0x08);
    HDMI_WriteI2C_Byte(0x3a,0x10);
    HDMI_WriteI2C_Byte(0x3b,0x20); //MIPIRX PCR de mode delay offset2

    HDMI_WriteI2C_Byte(0x3f,0x04); //PCR de mode step0 setting
    HDMI_WriteI2C_Byte(0x40,0x08);
    HDMI_WriteI2C_Byte(0x41,0x10);
    HDMI_WriteI2C_Byte(0x42,0x20);
    HDMI_WriteI2C_Byte(0x2b,0xA0);
HDMI_WriteI2C_Byte(0xff,0xd0); 	
    HDMI_WriteI2C_Byte(0x0c,0x60);  //fifo position
	HDMI_WriteI2C_Byte(0x1c,0x60);  //fifo position
	HDMI_WriteI2C_Byte(0x24,0x70);  //pcr mode( de hs vs)
			
	HDMI_WriteI2C_Byte(0x2d,0x30); //M up limit
	HDMI_WriteI2C_Byte(0x31,0x0a); //M down limit

	/*stage1 hs mode*/
	HDMI_WriteI2C_Byte(0x25,0xf0);  //line limit
	HDMI_WriteI2C_Byte(0x2a,0x30);  //step in limit
	

	

 
    #endif
    
    if (g_stRxVidTiming.ulPclk_Khz < 44000)
    {
        #if MIPIRX_CLK_BURST == ENABLED
//        HDMI_WriteI2C_Byte(0x0a,0x42);
        HDMI_WriteI2C_Byte(0x0c,0x40); //[7:0]rgd_vsync_dly(sram rd delay)
        HDMI_WriteI2C_Byte(0x1b,0x00); //pcr wr dly[15:0]
        HDMI_WriteI2C_Byte(0x1c,0x40); //pcr wr dly[7:0]
        #else
        HDMI_WriteI2C_Byte(0x0c,0x60); //[7:0]rgd_vsync_dly(sram rd delay)
        HDMI_WriteI2C_Byte(0x1b,0x00); //pcr wr dly[15:0]
        HDMI_WriteI2C_Byte(0x1c,0x60); //pcr wr dly[7:0]
        #endif
    }
    else 
    {
        HDMI_WriteI2C_Byte(0x0c,0x60); //[7:0]rgd_vsync_dly(sram rd delay)
        HDMI_WriteI2C_Byte(0x1b,0x00); //pcr wr dly[15:0]
        HDMI_WriteI2C_Byte(0x1c,0x60); //pcr wr dly[7:0]
    }
    
    HDMI_WriteI2C_Byte(0xff,0x81);     
    HDMI_WriteI2C_Byte(0x09,0xdb);
    HDMI_WriteI2C_Byte(0x09,0xdf); //pcr rst

    HDMI_WriteI2C_Byte(0xff,0xd0);
    HDMI_WriteI2C_Byte(0x08,0x80);
    HDMI_WriteI2C_Byte(0x08,0x00);
    usleep(10*1000);
    ucPcr_Cal_Cnt = 0;
    
    do
    {
        usleep(100*1000);
        ucPcr_Cal_Cnt++;
        printf("\nPCR unstable M = 0x%02x",(HDMI_ReadI2C_Byte(0x94)&0x7F));
    }while((ucPcr_Cal_Cnt < 10) && ((HDMI_ReadI2C_Byte(0x87) & 0x18) != 0x18));

    if((HDMI_ReadI2C_Byte(0x87) & 0x18) != 0x18)
    {
        printf("\nLT9211C pcr unstable");
        ucRtn = FAIL;
    }
    
    return ucRtn;
}

u8 Drv_MipiRx_VidFmtUpdate()
{
    u8 ucRxFmt;
    
    ucRxFmt = g_stMipiRxVidTiming_Get.ucFmt;
    HDMI_WriteI2C_Byte(0xff,0xd0);
    g_stMipiRxVidTiming_Get.ucFmt = (HDMI_ReadI2C_Byte(0x84) & 0x0f);
    
    if (ucRxFmt != g_stMipiRxVidTiming_Get.ucFmt)
    {
        return true;
    }
    else
    {
        return false;
    }
}


void Drv_MipiRx_HsSettleSet(void)
{
    if((g_stMipiRxVidTiming_Get.ucLane0SetNum > 0x10) && (g_stMipiRxVidTiming_Get.ucLane0SetNum < 0x50))
    {
        //shikeDebug
        //printf("\nSet Mipi Rx Settle: 0x%02x", (g_stMipiRxVidTiming_Get.ucLane0SetNum - 5));
        HDMI_WriteI2C_Byte(0xff,0xd0);
        HDMI_WriteI2C_Byte(0x02,(g_stMipiRxVidTiming_Get.ucLane0SetNum - 5));
    }
    else
    {
        //printf("\nSet Mipi Rx Settle: 0x03"); //mipi rx cts test need settle 0x0e
        HDMI_WriteI2C_Byte(0xff,0xd0);
        HDMI_WriteI2C_Byte(0x02,0x03);
    }
}

void Drv_MipiRx_SotGet()
{
    HDMI_WriteI2C_Byte(0xff,0xd0);
    g_stMipiRxVidTiming_Get.ucLane0SetNum  = HDMI_ReadI2C_Byte(0x88);
    g_stMipiRxVidTiming_Get.ucLane0SotData = HDMI_ReadI2C_Byte(0x89);
    g_stMipiRxVidTiming_Get.ucLane1SetNum  = HDMI_ReadI2C_Byte(0x8a);
    g_stMipiRxVidTiming_Get.ucLane1SotData = HDMI_ReadI2C_Byte(0x8b);
    g_stMipiRxVidTiming_Get.ucLane2SetNum  = HDMI_ReadI2C_Byte(0x8c);
    g_stMipiRxVidTiming_Get.ucLane2SotData = HDMI_ReadI2C_Byte(0x8d);
    g_stMipiRxVidTiming_Get.ucLane3SetNum  = HDMI_ReadI2C_Byte(0x8e);
    g_stMipiRxVidTiming_Get.ucLane3SotData = HDMI_ReadI2C_Byte(0x8f);

    // shikeDebug
    //printf("\nSot Num = 0x%02x, 0x%02x, 0x%02x, 0x%02x", g_stMipiRxVidTiming_Get.ucLane0SetNum,g_stMipiRxVidTiming_Get.ucLane1SetNum,
    //                                                                g_stMipiRxVidTiming_Get.ucLane2SetNum,g_stMipiRxVidTiming_Get.ucLane3SetNum);
    //printf("\nSot Dta = 0x%02x, 0x%02x, 0x%02x, 0x%02x", g_stMipiRxVidTiming_Get.ucLane0SotData,g_stMipiRxVidTiming_Get.ucLane1SotData,
    //                                                                g_stMipiRxVidTiming_Get.ucLane2SotData,g_stMipiRxVidTiming_Get.ucLane3SotData); 
}

void Drv_MipiRx_HactGet()
{
    HDMI_WriteI2C_Byte(0xff,0xd0);
    g_stMipiRxVidTiming_Get.usVact = (HDMI_ReadI2C_Byte(0x85) << 8) +HDMI_ReadI2C_Byte(0x86);
    g_stMipiRxVidTiming_Get.ucFmt  = (HDMI_ReadI2C_Byte(0x84) & 0x0f);
    g_stMipiRxVidTiming_Get.ucPa_Lpn = HDMI_ReadI2C_Byte(0x9c);
    g_stMipiRxVidTiming_Get.uswc = (HDMI_ReadI2C_Byte(0x82) << 8) + HDMI_ReadI2C_Byte(0x83); //

    switch (g_stMipiRxVidTiming_Get.ucFmt)
    {
        case 0x01: //DSI-YUV422-10bpc
        case 0x0e: //CSI-YUV422-10bpc
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 2.5; //wc = hact * 20bpp/8
        break;
        case 0x02: //DSI-YUV422-12bpc
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 3; //wc = hact * 24bpp/8
        break;
        case 0x03: //YUV422-8bpc
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 2; //wc = hact * 16bpp/8
        break; 
        case 0x04: //RGB10bpc
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 3.75; //wc = hact * 30bpp/8
        break;
        case 0x05: //RGB12bpc
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 4.5; //wc = hact * 36bpp/8
        break;
        case 0x06: //YUV420-8bpc
        case 0x0a: //RGB8bpc
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 3; //wc = hact * 24bpp/8
        break;
        case 0x07: //RGB565
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 2; //wc = hact * 16bpp/8
        break;
        case 0x08: //RGB6bpc
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc/ 2.25; //wc = hact * 18bpp/8
        break;
        case 0x09: //RGB6bpc_losely
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 3; //wc = hact * 24bpp/8
        break;
        case 0x0b: //RAW8
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 1; //wc = hact * 8bpp/8
        break;
        case 0x0c: //RAW10
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 1.25; //wc = hact * 10bpp/8
        break;
        case 0x0d: //RAW12
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 1.5; //wc = hact * 12bpp/8
        break;
        default: 
            g_stMipiRxVidTiming_Get.usHact = g_stMipiRxVidTiming_Get.uswc / 3; //wc = hact * 24bpp/8
        break;
    }
    
}

u8 Drv_MipiRx_VidTiming_Get(void)
{
    Drv_MipiRx_SotGet();
    Drv_MipiRx_HsSettleSet();
    Drv_MipiRx_HactGet();
    if((g_stMipiRxVidTiming_Get.usHact < 400) || (g_stMipiRxVidTiming_Get.usVact < 400))
    {
        ////LTLog(LOG_ERROR,"RX No Video Get");
        return FAIL;
    }
    else
    {
        //printf("\nhact = %d",g_stMipiRxVidTiming_Get.usHact);
        //printf("\nvact = %d",g_stMipiRxVidTiming_Get.usVact);    
        //printf("\nfmt = 0x%02x", g_stMipiRxVidTiming_Get.ucFmt);
        //printf("\npa_lpn = 0x%02x", g_stMipiRxVidTiming_Get.ucPa_Lpn);
        return SUCCESS;
    }
}

u8 Drv_MipiRx_VidTiming_Sel()
{	
    u8 rtn = false;
    u8 uci;
    u8 ucResolutionnum = 0;
    
    ucResolutionnum = sizeof(resolution) / sizeof(resolution[0]);
    
    for (uci = 0; uci < ucResolutionnum; uci++)
    {
        if ((g_stMipiRxVidTiming_Get.usHact == resolution[uci].usHact ) && 
            ( g_stMipiRxVidTiming_Get.usVact == resolution[uci].usVact ))
        {
            g_stMipiRxVidTiming_Get.ucFrameRate = Drv_VidChk_FrmRt_Get();
            printf("\nFrameRate = %d", g_stMipiRxVidTiming_Get.ucFrameRate);
            if ((g_stMipiRxVidTiming_Get.ucFrameRate > (resolution[uci].ucFrameRate - 3)) && 
               (g_stMipiRxVidTiming_Get.ucFrameRate < (resolution[uci].ucFrameRate + 3)))
            {
                g_stRxVidTiming.usVtotal = resolution[uci].usVtotal;
                g_stRxVidTiming.usVact   = resolution[uci].usVact;
                g_stRxVidTiming.usVs     = resolution[uci].usVs;
                g_stRxVidTiming.usVfp    = resolution[uci].usVfp;
                g_stRxVidTiming.usVbp    = resolution[uci].usVbp;
                
                g_stRxVidTiming.usHtotal = resolution[uci].usHtotal;
                g_stRxVidTiming.usHact   = resolution[uci].usHact;
                g_stRxVidTiming.usHs     = resolution[uci].usHs;
                g_stRxVidTiming.usHfp    = resolution[uci].usHfp;
                g_stRxVidTiming.usHbp    = resolution[uci].usHbp;
                g_stRxVidTiming.ulPclk_Khz = (u32)((u32)(resolution[uci].usHtotal) * (resolution[uci].usVtotal) * (resolution[uci].ucFrameRate) / 1000);

                Drv_MipiRx_VidTiming_Set();
                printf("\nVideo Timing Set %dx%d_%d",g_stRxVidTiming.usHact,g_stRxVidTiming.usVact,g_stMipiRxVidTiming_Get.ucFrameRate);
                rtn = true;
                break;
            }
        }
    }
    return rtn;
 
}

void Drv_MipiRx_VidTiming_Set()
{
    HDMI_WriteI2C_Byte(0xff,0xd0);
    HDMI_WriteI2C_Byte(0x0d,(u8)(g_stRxVidTiming.usVtotal >> 8));     //vtotal[15:8]
    HDMI_WriteI2C_Byte(0x0e,(u8)(g_stRxVidTiming.usVtotal));          //vtotal[7:0]
    HDMI_WriteI2C_Byte(0x0f,(u8)(g_stRxVidTiming.usVact >> 8));       //vactive[15:8]
    HDMI_WriteI2C_Byte(0x10,(u8)(g_stRxVidTiming.usVact));            //vactive[7:0]
    HDMI_WriteI2C_Byte(0x15,(u8)(g_stRxVidTiming.usVs));              //vs[7:0]
    HDMI_WriteI2C_Byte(0x17,(u8)(g_stRxVidTiming.usVfp >> 8));        //vfp[15:8]
    HDMI_WriteI2C_Byte(0x18,(u8)(g_stRxVidTiming.usVfp));             //vfp[7:0]    

    HDMI_WriteI2C_Byte(0x11,(u8)(g_stRxVidTiming.usHtotal >> 8));     //htotal[15:8]
    HDMI_WriteI2C_Byte(0x12,(u8)(g_stRxVidTiming.usHtotal));          //htotal[7:0]
    HDMI_WriteI2C_Byte(0x13,(u8)(g_stRxVidTiming.usHact >> 8));       //hactive[15:8]
    HDMI_WriteI2C_Byte(0x14,(u8)(g_stRxVidTiming.usHact));            //hactive[7:0]
    HDMI_WriteI2C_Byte(0x4c,(u8)(g_stRxVidTiming.usHs >> 8));         //hs[15:8]
    HDMI_WriteI2C_Byte(0x16,(u8)(g_stRxVidTiming.usHs));              //hs[7:0]
    HDMI_WriteI2C_Byte(0x19,(u8)(g_stRxVidTiming.usHfp >> 8));        //hfp[15:8]
    HDMI_WriteI2C_Byte(0x1a,(u8)(g_stRxVidTiming.usHfp));             //hfp[7:0]
}

u8 Drv_MipiRx_VidFmt_Get(IN u8 VidFmt)
{
    u8 ucRxVidFmt;
    
    switch (VidFmt)
    {
        case 0x01: //DSI-YUV422-10bpc
            ucRxVidFmt = YUV422_10bit;
        break;
        case 0x02: //DSI-YUV422-12bpc
            ucRxVidFmt = YUV422_12bit;
        break;
        case 0x03: //YUV422-8bpc
            ucRxVidFmt = YUV422_8bit;
        break;
        case 0x04: //RGB30bpp
            ucRxVidFmt = RGB_10Bit;
        break;
        case 0x05: //RGB36bpp
            ucRxVidFmt = RGB_12Bit;
        break;
        case 0x06: //YUV420-8bpc
            ucRxVidFmt = YUV420_8bit;
        break;
        case 0x07: //RGB565
        break;
        case 0x08: //RGB666
            ucRxVidFmt = RGB_6Bit;
        break;
        case 0x09: //DSI-RGB6L
        break;
        case 0x0a: //RGB888
            ucRxVidFmt = RGB_8Bit;
        break;
        case 0x0b: //RAW8
        break;
        case 0x0c: //RAW10
        break;
        case 0x0d: //RAW12
        break;
        case 0x0e: //CSI-YUV422-10
            ucRxVidFmt = YUV422_10bit;
        break;
        default:
            ucRxVidFmt = RGB_8Bit;
        break;    
    }

    //printf("\nMipiRx Input Format: %s",g_szStrRxFormat[VidFmt]);
    return ucRxVidFmt;
}

void Drv_MipiRx_InputSel()
{
    HDMI_WriteI2C_Byte(0xff,0xd0);
    #if (MIPIRX_INPUT_SEL == MIPI_CSI)
    HDMI_WriteI2C_Byte(0x04,0x10); //[4]1: CSI enable
    HDMI_WriteI2C_Byte(0x21,0xc6); //[7](dsi: hsync_level(for pcr adj) = hsync_level; csi:hsync_level(for pcr adj) = de_level)
    //printf("\nMipi CSI Input");
    #else 
    HDMI_WriteI2C_Byte(0x04,0x00); //[4]0: DSI enable
    HDMI_WriteI2C_Byte(0x21,0x46); //[7](dsi: hsync_level(for pcr adj) = hsync_level; csi:hsync_level(for pcr adj) = de_level)
    //printf("\nMipi DSI Input");
    #endif
}

void Drv_MipiRx_LaneSet(void)
{
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x3f,0x08); //MLRX HS/LP control conmand enable
    HDMI_WriteI2C_Byte(0x40,0x04); //[2:0]pa_ch0_src_sel ch4 data
    HDMI_WriteI2C_Byte(0x41,0x03); //[2:0]pa_ch1_src_sel ch3 data
    HDMI_WriteI2C_Byte(0x42,0x02); //[2:0]pa_ch2_src_sel ch2 data
    HDMI_WriteI2C_Byte(0x43,0x01); //[2:0]pa_ch3_src_sel ch1 data

    HDMI_WriteI2C_Byte(0x45,0x04); //[2:0]pb_ch0_src_sel ch9 data
    HDMI_WriteI2C_Byte(0x46,0x03); //[2:0]pb_ch1_src_sel ch8 data
    HDMI_WriteI2C_Byte(0x47,0x02); //[2:0]pb_ch2_src_sel ch7 data
    HDMI_WriteI2C_Byte(0x48,0x01); //[2:0]pb_ch3_src_sel ch6 data
    
    HDMI_WriteI2C_Byte(0x44,0x00); //[6]mlrx port A output select port A;[2:0]pa_ch4_src_sel ch0 data
    HDMI_WriteI2C_Byte(0x49,0x00); //[6]mlrx port B output select port A;[2:0]pb_ch4_src_sel ch5 data

    #if MIPIRX_PORT_SEL == PORTB
    HDMI_WriteI2C_Byte(0x44,0x40); //[6]mlrx port A output select port B;[2:0]pa_ch4_src_sel ch0 data
    HDMI_WriteI2C_Byte(0x49,0x00); //[6]mlrx port B output select port B;[2:0]pb_ch4_src_sel ch5 data
    #endif

}

void Drv_MipiRxClk_Sel(void)
{
    /* CLK sel */
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0xe9,0x88); //sys clk sel from XTAL
    
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x80,0x51); //[7:6]rx sram rd clk src sel ad dessc pcr clk
                                   //[5:4]rx sram wr clk src sel mlrx bytr clk
                                   //[1:0]video check clk sel from desscpll pix clk
    #if MIPIRX_PORT_SEL == PORTA
    HDMI_WriteI2C_Byte(0x81,0x10); //[5]0: mlrx byte clock select from ad_mlrxa_byte_clk
                                   //[4]1: rx output pixel clock select from ad_desscpll_pix_clk
    #elif MIPIRX_PORT_SEL == PORTB
    HDMI_WriteI2C_Byte(0x81,0x30); //[5]1: mlrx byte clock select from ad_mlrxb_byte_clk
                                   //[4]1: rx output pixel clock select from ad_desscpll_pix_clk
    #endif
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0x32,0x03); //video check frame cnt set: 3 frame
}
void Drv_MipiRx_VidChkDebug(void)
{
    u16 ushact, usvact;
    u16 ushs, usvs;
    u16 ushbp, usvbp;
    u16 ushtotal, usvtotal;
    u16 ushfp, usvfp;
	

	usleep(100*1000);
	HDMI_WriteI2C_Byte(0xff,0x86);
    
	
	
	usvs = HDMI_ReadI2C_Byte(0x52); //[15:8]
	usvs = (usvs <<8 ) + HDMI_ReadI2C_Byte(0x53);

	ushs = HDMI_ReadI2C_Byte(0x50);
    ushs = (ushs << 8) + HDMI_ReadI2C_Byte(0x51);
	
	usvbp = HDMI_ReadI2C_Byte(0x57);
    usvfp = HDMI_ReadI2C_Byte(0x5b);

	ushbp = HDMI_ReadI2C_Byte(0x54);
	ushbp = (ushbp << 8) + HDMI_ReadI2C_Byte(0x55);

	ushfp = HDMI_ReadI2C_Byte(0x58);
	ushfp = (ushfp << 8) + HDMI_ReadI2C_Byte(0x59);

	usvtotal = HDMI_ReadI2C_Byte(0x62);
	usvtotal = (usvtotal << 8) + HDMI_ReadI2C_Byte(0x63);

	ushtotal = HDMI_ReadI2C_Byte(0x60);
	ushtotal = (ushtotal << 8) + HDMI_ReadI2C_Byte(0x61);

	usvact = HDMI_ReadI2C_Byte(0x5e);
	usvact = (usvact << 8)+ HDMI_ReadI2C_Byte(0x5f);

	ushact = HDMI_ReadI2C_Byte(0x5c);
	ushact = (ushact << 8) + HDMI_ReadI2C_Byte(0x5d);

	
	
    //printf("\nmipi input hfp, hs, hbp, hact, htotal = %d %d %d %d %d",ushfp, ushs, ushbp, ushact, ushtotal);
	//printf("\nmipi input vfp, vs, vbp, vact, vtotal = %d %d %d %d %d",usvfp, usvs, usvbp, usvact, usvtotal);
}
void Drv_MipiRx_PhyPowerOn(void)
{
    HDMI_WriteI2C_Byte(0xff,0xd0);
    HDMI_WriteI2C_Byte(0x00,(HDMI_ReadI2C_Byte(0x00) | MIPIRX_LANE_NUM));    // 0: 4 Lane / 1: 1 Lane / 2 : 2 Lane / 3: 3 Lane

    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x01,0x11); //MIPI RX portA & B disable

    #if MIPIRX_PORT_SEL == PORTA
    HDMI_WriteI2C_Byte(0x18,0x48); //portA clk delay select 0
    HDMI_WriteI2C_Byte(0x01,0x91); //MIPI RX portA enable
    HDMI_WriteI2C_Byte(0x02,0x00); //[5][1]:0 mipi mode, no swap
    HDMI_WriteI2C_Byte(0x03,0xEE); //port A & B eq current reference   
    HDMI_WriteI2C_Byte(0x09,0x21); //[3]0: select link clk from port-A, [1]0: mlrx_clk2pll disable
    HDMI_WriteI2C_Byte(0x04,0x44);
    HDMI_WriteI2C_Byte(0x05,0xc4); //port A clk lane eq sel
    HDMI_WriteI2C_Byte(0x06,0x44);
    HDMI_WriteI2C_Byte(0x13,0x0c); //MIPI port A clk lane rterm & high speed en
    
    #if MIPIRX_CLK_BURST == ENABLED
    HDMI_WriteI2C_Byte(0x13,0x00); //MIPI port A clk lane rterm & high speed en
    #endif
    //printf("\nMIPI Input PortA");
    #elif MIPIRX_PORT_SEL == PORTB
    HDMI_WriteI2C_Byte(0x1A, 0x01); //portB clk  delay select 0
    HDMI_WriteI2C_Byte(0x01,0x19); //MIPI RX portB enable
    HDMI_WriteI2C_Byte(0x02,0x00); //[5][1]:0 mipi mode, no swap
    HDMI_WriteI2C_Byte(0x03,0xEE); //port A & B eq current reference
    HDMI_WriteI2C_Byte(0x06,0x44);
    HDMI_WriteI2C_Byte(0x07,0x4c); //port B clk lane eq sel
    HDMI_WriteI2C_Byte(0x08,0x44);
    HDMI_WriteI2C_Byte(0x09,0x29); //[3]1: select link clk from port-B, [1]0: mlrx_clk2pll disable
    HDMI_WriteI2C_Byte(0x14,0x03); //Port-B clk lane software enable
    
    #if MIPIRX_CLK_BURST == ENABLED
    HDMI_WriteI2C_Byte(0x14,0x00); //MIPI port A clk lane rterm & high speed en
    #endif
    //printf("\nMIPI Input PortB");
    #else
    HDMI_WriteI2C_Byte(0x01,0x99); //MIPI RX portB enable
    HDMI_WriteI2C_Byte(0x02,0x00); //[5][1]:0 mipi mode, no swap
    HDMI_WriteI2C_Byte(0x03,0xee); //port A & B eq current reference 
    HDMI_WriteI2C_Byte(0x09,0x29); //[3]1: select link clk from port-B, [1]0: mlrx_clk2pll disable
    HDMI_WriteI2C_Byte(0x13,0x0c); //MIPI port A clk lane rterm & high speed en
    HDMI_WriteI2C_Byte(0x14,0x03); //Port-B clk lane software enable
    
    #if MIPIRX_CLK_BURST == ENABLED
    HDMI_WriteI2C_Byte(0x13,0x00); //MIPI port A clk lane rterm use hardware mode
    HDMI_WriteI2C_Byte(0x14,0x00); //Port-B clk lane use hardware mode
    #endif
    #endif

    HDMI_WriteI2C_Byte(0xff,0xd0);
    HDMI_WriteI2C_Byte(0x01,0x00); //mipi rx data lane term enable time: 39ns;
    HDMI_WriteI2C_Byte(0x02,0x0e); //mipi rx hs settle time defult set: 0x05;
    HDMI_WriteI2C_Byte(0x05,0x00); //mipi rx lk lane term enable time: 39ns;
    HDMI_WriteI2C_Byte(0x0a,0x59);
    HDMI_WriteI2C_Byte(0x0b,0x20);
    
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x09,0xde); //mipi rx dphy reset
    HDMI_WriteI2C_Byte(0x09,0xdf); //mipi rx dphy release
}


#endif