#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
  * @project: LT9211C
  * @file: ModPattern.c
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#include	"ModPattern.h"


#if (LT9211C_MODE_SEL  == PATTERN_OUT)
#if TX_VID_PATTERN_SEL == PTN_720x288_50
struct ChipRxVidTiming ptn_timing   = {12,   64,    68,    720,    864,    2,   2,  20,   288,   312,  13500};
#endif
#if TX_VID_PATTERN_SEL == PTN_640x480_15
struct ChipRxVidTiming ptn_timing   = {24,  96,  40,   640,    800,    33,  2,   10,  480,  525,  6250};
#endif
#if TX_VID_PATTERN_SEL == PTN_640x480_60
struct ChipRxVidTiming ptn_timing   = {24,  96,  40,   640,    800,    33,  2,   10,  480,  525,  25000};
#endif
#if TX_VID_PATTERN_SEL == PTN_720x480_60
struct ChipRxVidTiming ptn_timing   = {16,  62,  60,   720,    858,    9,   6,   30,  480,  525,  27000};
#endif
#if TX_VID_PATTERN_SEL == PTN_1280x720_60
struct ChipRxVidTiming ptn_timing   = {110, 40,  220,  1280,   1650,   5,   5,   20,  720,  750,  74250};
#endif
#if TX_VID_PATTERN_SEL == PTN_1920x1080_30
struct ChipRxVidTiming ptn_timing   = {88,  44,  148,  1920,   2200,   4,   5,   36,  1080, 1125, 74250};
#endif
#if TX_VID_PATTERN_SEL == PTN_1920x1080_60
struct ChipRxVidTiming ptn_timing   = {88,  44,  148,  1920,   2200,   4,   5,   36,  1080, 1125, 148500};
#endif
#if TX_VID_PATTERN_SEL == PTN_1920x1080_90
struct ChipRxVidTiming ptn_timing   = {88,  44,  148,  1920,   2200,   4,   5,   36,  1080, 1125, 222750};
#endif
#if TX_VID_PATTERN_SEL == PTN_1920x1080_100
struct ChipRxVidTiming ptn_timing   = {88,  44,  148,  1920,   2200,   4,   5,   36,  1080, 1125, 247500};
#endif
#if TX_VID_PATTERN_SEL == PTN_1920x720_60
struct ChipRxVidTiming ptn_timing   = {48,  32,  80,   1920,   2080,   5,   5,   20,  720,  750,  93600 };
#endif
#if TX_VID_PATTERN_SEL == PTN_2560x1440_60
struct ChipRxVidTiming ptn_timing   = {48,  32,  80,   2560,   2720,   3,   5,   33,  1440, 1481, 241700 };
#endif
#if TX_VID_PATTERN_SEL == PTN_3840x2160_30
struct ChipRxVidTiming ptn_timing   = {176, 88,  296,  3840,   4400, 8,   10,  72,  2160, 2250, 297000};
#endif
#if TX_VID_PATTERN_SEL == PTN_1024x600_60
struct ChipRxVidTiming ptn_timing   = {160, 20, 140,1024,  1344,  12,  3,  20,  600, 635  , 51688};
#endif
 #if TX_VID_PATTERN_SEL == PTN_440x1920_60
struct ChipRxVidTiming ptn_timing   = {53, 4, 53,440, 550,  16,  4,  12,  1920, 1952  , 65000};
#endif                                                   //hfp    hs     hbp     hact     htotal   vfp   vs   vbp   vact    vtotal 
struct video_pattern_timing pattern_640x480_15Hz   = {24,    96,    40,     640,     800,     33,   2,   10,   480,    525,     15};
struct video_pattern_timing pattern_640x480_60Hz   = {24,    96,    40,     640,     800,     33,   2,   10,   480,    525,     60};
struct video_pattern_timing pattern_720x480_60Hz   = {16,    62,    60,     720,     858,     9,    6,   30,   480,    525,     60};
struct video_pattern_timing pattern_1280x720_60Hz  = {110,   40,    220,    1280,    1650,    5,    5,   20,   720,    750,     60};
struct video_pattern_timing pattern_1920x720_60Hz  = {48,    32,    80,     1920,    2080,    5,    5,   20,   720,    750,     60};
struct video_pattern_timing pattern_1920x1080_30Hz = {88,    44,    148,    1920,    2200,    4,    5,   36,   1080,   1125,    30};
struct video_pattern_timing pattern_1920x1080_60Hz = {88,    44,    148,    1920,    2200,    4,    5,   36,   1080,   1125,    60};
struct video_pattern_timing pattern_1920x1080_90Hz = {88,    44,    148,    1920,    2200,    4,    5,   36,   1080,   1125,    90};
struct video_pattern_timing pattern_1920x1080_100Hz= {88,    44,    148,    1920,    2200,    4,    5,   36,   1080,   1125,   100};
struct video_pattern_timing pattern_2560x1440_60Hz = {48,    32,    80,     2560,    2720,    3,    5,   33,   1440,   1481,    60};
struct video_pattern_timing pattern_3840x2160_30Hz = {176,   88,    296,    3840,    4400,    8,   10,  72,   2160,   2250,    30};
struct video_pattern_timing pattern_720x288_50Hz = {12,   64,    68,    720,    864,    2,   2,  20,   288,   312,    50};
struct video_pattern_timing pattern_1024x600_60Hz = {160, 20, 140,1024,  1344,  12,  3,  20,  600, 635,    60};

struct video_pattern_timing pattern_440x1920_60Hz = {53, 4, 53,440, 550,  16,  4,  12,  1920, 1952,    60};


void Mod_SystemInt(void)
{
    /* system clock init */		   
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0xe9,0x88); //sys clk sel from XTAL
}

void Mod_DesscPll_ForPtn(struct ChipRxVidTiming *ptn_timing)
{
    u32 pclk_khz;
    u8 desscpll_pixck_div;
    u32 pcr_m, pcr_k;

    pclk_khz = ptn_timing->ulPclk_Khz; 
    printf("\npclk_khz: %ld",pclk_khz);
    
  	/* dessc pll */
    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x26,0x20); //[7:6]desscpll reference select Xtal clock as reference
                                   //[4]1'b0: dessc-pll power down
	
    
    if(pclk_khz < 11000)
	{
        HDMI_WriteI2C_Byte(0x2f,0x07);
		desscpll_pixck_div = 16;
        HDMI_WriteI2C_Byte(0x2c,0x02);    //lowf_pix_div = 4
        pclk_khz = 4 * pclk_khz;
	}
    else if(pclk_khz < 22000)
	{
        HDMI_WriteI2C_Byte(0x2f,0x07);
		desscpll_pixck_div = 16;
        HDMI_WriteI2C_Byte(0x2c,0x01);    //lowf_pix_div = 2
        pclk_khz = 2 * pclk_khz;
	}
    else if(pclk_khz < 44000)
	{
        HDMI_WriteI2C_Byte(0x2f,0x07);
		desscpll_pixck_div = 16;
	}
	else if(pclk_khz < 88000)
	{
	  	HDMI_WriteI2C_Byte(0x2f,0x06);
		desscpll_pixck_div = 8;
	}
	else if(pclk_khz < 176000)
	{
	  	HDMI_WriteI2C_Byte(0x2f,0x05);
		desscpll_pixck_div = 4;
	}
	else if(pclk_khz < 352000)
	{
	  	HDMI_WriteI2C_Byte(0x2f,0x04);
		desscpll_pixck_div = 2;
	}
	else
	{
        HDMI_WriteI2C_Byte(0x2f,0x04);
		desscpll_pixck_div = 2;
	}

    pcr_m = (pclk_khz * desscpll_pixck_div) /25;
	pcr_k = pcr_m%1000;
	pcr_m = pcr_m/1000;

	pcr_k <<= 14;
	
	//pixel clk
 	HDMI_WriteI2C_Byte(0xff,0xd0); //pcr
	HDMI_WriteI2C_Byte(0x2d,0x7f);
	HDMI_WriteI2C_Byte(0x31,0x00);

	HDMI_WriteI2C_Byte(0x26,0x80|((u8)pcr_m));
	HDMI_WriteI2C_Byte(0x27,(u8)((pcr_k>>16)&0xff)); //K
	HDMI_WriteI2C_Byte(0x28,(u8)((pcr_k>>8)&0xff)); //K
	HDMI_WriteI2C_Byte(0x29,(u8)(pcr_k&0xff)); //K

    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x03,0xfe); //desscpll rst
    usleep(10*1000);
    HDMI_WriteI2C_Byte(0x03,0xff);
}

void Mod_VidChkDebug_ForPtn(void)
{
    u16 ushact, usvact;
    u16 ushs, usvs;
    u16 ushbp, usvbp;
    u16 ushtotal, usvtotal;
    u16 ushfp, usvfp;
	u8 ucsync_polarity;
	u8 ucframerate;

	HDMI_WriteI2C_Byte(0xff,0x86);
	HDMI_WriteI2C_Byte(0x3f,0x03); //video check clk src sel : pattern
	usleep(100*1000);
    ucframerate = Drv_VidChk_FrmRt_Get();
    printf("\nucframerate: %d",ucframerate);
	HDMI_WriteI2C_Byte(0xff,0x86);
	ucsync_polarity = HDMI_ReadI2C_Byte(0x4f);
	
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

	
	printf("\nsync_polarity = 0x%02x", ucsync_polarity);

     if ((ushact == pattern_440x1920_60Hz.hact ) &&( usvact == pattern_440x1920_60Hz.vact ))
    {
        printf("\nVideo Pattern Set pattern_440x1920_60Hz");
    }
    else if ((ushact == pattern_720x288_50Hz.hact ) &&( usvact == pattern_720x288_50Hz.vact ))
    {
        printf("\nVideo Pattern Set pattern_720x288_50Hz");
    }
    else if ((ushact == pattern_640x480_60Hz.hact ) &&( usvact == pattern_640x480_60Hz.vact ))
    {
        printf("\nVideo Pattern Set 640x480");
    }
    else if ((ushact == pattern_720x480_60Hz.hact ) &&(usvact == pattern_720x480_60Hz.vact ))
    {
        printf("\nVideo Pattern Set 720x480_60Hz");
    }
    else if ((ushact == pattern_1280x720_60Hz.hact ) &&(usvact == pattern_1280x720_60Hz.vact ))
    {
        printf("\nVideo Pattern Set 1280x720_60Hz");
    }
    else if ((ushact == pattern_1920x720_60Hz.hact ) &&(usvact == pattern_1920x720_60Hz.vact ))
    {
        printf("\nVideo Pattern Set 1920x720_60Hz");
    }
    else if ((ushact == pattern_1920x1080_60Hz.hact ) &&(usvact == pattern_1920x1080_60Hz.vact ))
    {
        if (ucframerate > (pattern_1920x1080_90Hz.ucFrameRate - 3) && ucframerate < (pattern_1920x1080_90Hz.ucFrameRate + 3))
        {
            printf("\nVideo Pattern Set 1920x1080_90Hz");
        }
        else if (ucframerate > (pattern_1920x1080_100Hz.ucFrameRate - 3) && ucframerate < (pattern_1920x1080_100Hz.ucFrameRate + 3))
        {
            printf("\nVideo Pattern Set 1920x1080_100Hz");
        }
        else
        {
            printf("\nVideo Pattern Set 1920x1080_60Hz");
        }
    }
    else if ((ushact == pattern_2560x1440_60Hz.hact ) &&(usvact == pattern_2560x1440_60Hz.vact ))
    {
        printf("\nVideo Pattern Set 2560x1440_60Hz");
    }
    else if ((ushact == pattern_3840x2160_30Hz.hact ) &&(usvact == pattern_3840x2160_30Hz.vact ))
    {
        printf("\nVideo Pattern Set 3840x2160_30Hz");
    }
    else 
    {
        printf("\nNo Video Set");
    }

    printf("\nhfp, hs, hbp, hact, htotal = %d %d %d %d %d",ushfp, ushs, ushbp, ushact, ushtotal);
	printf("\nvfp, vs, vbp, vact, vtotal = %d %d %d %d %d",usvfp, usvs, usvbp, usvact, usvtotal);
}


#if (TX_PATTERN_SRC_SEL  == MIPITX_PATTERN)

u8 ucTxLaneNum;
u8 b1MipiClockburst;
u32 ulMipiDataRate = 0;

void Mod_MipiTx_PtnTiming_Set(struct ChipRxVidTiming *ptn_timing)
{

    u16 ushss, usvss;
    /* CLK sel */
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x80,0x01); //[7:6]rx sram rd clk src sel ad dessc pcr clk
                                   //[5:4]rx sram wr clk src sel mlrx bytr clk
                                   //[1:0]video check clk sel from desscpll pix clk
    HDMI_WriteI2C_Byte(0x81,0x10); //[5]mlrx byte clock select from ad_mlrxb_byte_clk
                                   //[4]rx output pixel clock select from ad_desscpll_pix_clk
                                   
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x30,0x03); //Chip active RX select pattern
    
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0x25,0x11); //Test video pattern encode mode select
    HDMI_WriteI2C_Byte(0x26,0x04); //

    HDMI_WriteI2C_Byte(0x11,0x77); //pattern mode sel
    HDMI_WriteI2C_Byte(0x12,0xff); //Pattern data value set

#if MIPITX_PATTERN_COLORSPACE == YUV422
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0x85,0x10); //[4]1'b1:csc RGB convert to YCbCr enable
    HDMI_WriteI2C_Byte(0x86,0x40); //[6]1'b1:csc YUV444 convert to YUV422 enable
//    #if MIPITX_PATTERN_COLORSPACE == YUV420
//    HDMI_WriteI2C_Byte(0x86,0x44); //[6]1'b1:csc YUV444 convert to YUV422 enable
//    #endif
    
#endif
   
    ushss = ptn_timing->usHs + ptn_timing->usHbp;
    usvss = ptn_timing->usVs + ptn_timing->usVbp;
    
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0x1d,(u8)(ptn_timing->usVtotal >> 8));       //vtotal[15:8]
    HDMI_WriteI2C_Byte(0x1e,(u8)(ptn_timing->usVtotal));            //vtotal[7:0]
    HDMI_WriteI2C_Byte(0x19,(u8)(ptn_timing->usVact >> 8));         //vactive[15:8]
    HDMI_WriteI2C_Byte(0x1a,(u8)(ptn_timing->usVact));              //vactive[7:0]
    HDMI_WriteI2C_Byte(0x15,(u8)(usvss >> 8));                      //vss[15:8]
    HDMI_WriteI2C_Byte(0x16,(u8)(usvss));                           //vss[7:0]
    HDMI_WriteI2C_Byte(0x21, (u8)(ptn_timing->usVs));               //vs[7:0]
   
    HDMI_WriteI2C_Byte(0x1b,(u8)(ptn_timing->usHtotal >> 8));       //htotal[15:8]
    HDMI_WriteI2C_Byte(0x1c,(u8)(ptn_timing->usHtotal));            //htotal[7:0]
    HDMI_WriteI2C_Byte(0x17,(u8)(ptn_timing->usHact >> 8));         //hactive[15:8]
    HDMI_WriteI2C_Byte(0x18,(u8)(ptn_timing->usHact));              //hactive[7:0]
    HDMI_WriteI2C_Byte(0x13,(u8)(ushss >> 8));                      //hss[15:8]
    HDMI_WriteI2C_Byte(0x14,(u8)(ushss));                           //hss[7:0]
    HDMI_WriteI2C_Byte(0x1f,(u8)(ptn_timing->usHs >> 8));           //hs[15:8]
    HDMI_WriteI2C_Byte(0x20,(u8)(ptn_timing->usHs));                //hs[7:0]

}

void Mod_MipiTxpll_ForPtn(struct ChipRxVidTiming *ptn_timing)
{
	u8 ucloopx = 0;
	u8 ucSericlkDiv , ucPreDiv, ucDivSet;
	u32 ulMpiTXPhyClk;

    ucTxLaneNum = MIPITX_PATTERN_OUTPUT_LANE;
    b1MipiClockburst = MIPITX_PATTERN_CLOCK_BURST;
    
#if MIPITX_PATTERN_COLORSPACE == YUV422
    ulMipiDataRate = ptn_timing->ulPclk_Khz * (16 / ucTxLaneNum); //pix clk * 16(bpp) / (1(port) * 4(lane))//YUV422 data format
#elif MIPITX_PATTERN_COLORSPACE == RGB_666L
    ulMipiDataRate = ptn_timing->ulPclk_Khz * (24 / ucTxLaneNum); //pix clk * 24(bpp) / (1(port) * 4(lane))//RGB666L data format
#elif MIPITX_PATTERN_COLORSPACE == RGB_666
    ulMipiDataRate = ptn_timing->ulPclk_Khz * (18 / ucTxLaneNum); //pix clk * 18(bpp) / (1(port) * 4(lane))//RGB666 data format
#elif MIPITX_PATTERN_COLORSPACE == RGB_565
    ulMipiDataRate = ptn_timing->ulPclk_Khz * (16 / ucTxLaneNum); //pix clk * 16(bpp) / (1(port) * 4(lane))//RGB565 data format
#elif MIPITX_PATTERN_COLORSPACE == RGB_888
    ulMipiDataRate = ptn_timing->ulPclk_Khz * (24 / ucTxLaneNum); //pix clk * 24(bpp) / (1(port) * 4(lane))//RGB888 data format
#endif   
    if(b1MipiClockburst)
    {
        ulMipiDataRate += 120000;

    }
    else
    {
        //ulMipiDataRate += 100000;
    }

    if(ulMipiDataRate <= 1500000)
    {
        ulMpiTXPhyClk = ulMipiDataRate;
        HDMI_WriteI2C_Byte(0xff,0x82);
        HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) & BIT4_0));
    }
    else
    {
        ulMpiTXPhyClk = ulMipiDataRate / 2;
    }

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

    printf("\nulMipiDataRate: %ld, ulMpiTXPhyClk: %ld, ucSericlkDiv: %d, ucPreDiv: %d, ucDivSet: %d",
                      ulMipiDataRate, ulMpiTXPhyClk, ucSericlkDiv, ucPreDiv, ucDivSet);

	HDMI_WriteI2C_Byte(0xff,0x81);
	HDMI_WriteI2C_Byte(0x0c,0xfe); //txpll reset
	usleep(5*1000);
	HDMI_WriteI2C_Byte(0x0c,0xff); //txpll release

    do
    {
		HDMI_WriteI2C_Byte(0xff,0x87);
		HDMI_WriteI2C_Byte(0x0f,0x00);
		HDMI_WriteI2C_Byte(0x0f,0x01);
		usleep(20*1000);
		ucloopx++;
	}while((ucloopx < 3) && ((HDMI_ReadI2C_Byte(0x39) & 0x01) != 0x01));

    if(HDMI_ReadI2C_Byte(0x39) & 0x04)
    {
        printf("\nLT9211C tx pll lock");
    }
    else
    {
        printf("\nLT9211C tx pll unlocked");
    }
 	
}


void Mod_MipiTxPhy_ForPtn()
{
    HDMI_WriteI2C_Byte(0xff,0x82);
    #if MIPITX_PATTERN_OUTPUT_PORT == PORTB
    HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) | 0x02)); //portA disbale & B enable
    #else
    HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) | 0x03)); //portA & B enable
    #endif
    HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) | 0x80)); //mipitx phy use half speed byte clk

    #if MIPITX_PATTERN_OUTPUT_PORT == PORTA
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x4a,(HDMI_ReadI2C_Byte(0x4a) | BIT6_1)); //[6] porta sel src portb
    HDMI_WriteI2C_Byte(0x50,(HDMI_ReadI2C_Byte(0x50) | BIT6_1)); //[6] portb sel src portb
    #endif

    //mipi tx phy cts test require
if ((ulMipiDataRate > 1500000) || (ulMipiDataRate <= 1050000)) //mipitx D-Dphy cts 1.4.17 test fail when datarate less than 1.5Gbps if use those setting
    {
        HDMI_WriteI2C_Byte(0xff,0x82);
        HDMI_WriteI2C_Byte(0x37,0x11);
        HDMI_WriteI2C_Byte(0x39,0x32);
        HDMI_WriteI2C_Byte(0x3a,0xca);
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

void Mod_MipiTx_TimingSet_ForPtn(struct ChipRxVidTiming *ptn_timing)
{
    u16 ushss, usvss;
    u16 us3d_dly;

    ushss = ptn_timing->usHs + ptn_timing->usHbp;
    usvss = ptn_timing->usVs + ptn_timing->usVbp;
    
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x7e,(u8)(ptn_timing->usHact >> 8));
    HDMI_WriteI2C_Byte(0x7f,(u8)ptn_timing->usHact);
    HDMI_WriteI2C_Byte(0x7c,(u8)(ptn_timing->usVact >> 8));
    HDMI_WriteI2C_Byte(0x7d,(u8)ptn_timing->usVact);
    HDMI_WriteI2C_Byte(0x84,(u8)(HDMI_ReadI2C_Byte(0x84) | ((ushss >> 8) & 0xfc)));
    HDMI_WriteI2C_Byte(0x85,(u8)ushss);
    HDMI_WriteI2C_Byte(0x80,(u8)(usvss >> 8));
    HDMI_WriteI2C_Byte(0x81,(u8)usvss);

    //3d dly
    us3d_dly = ptn_timing->usHact / 4;
    HDMI_WriteI2C_Byte(0x7a,(u8)(us3d_dly >> 8));
    HDMI_WriteI2C_Byte(0x7b,(u8)us3d_dly);

    //hsync & vsync positive
    HDMI_WriteI2C_Byte(0x70,(HDMI_ReadI2C_Byte(0x70) & 0xfc));
}

void Mod_MipiTx_PhyTimingParaSet_ForPtn()
{
    u8 ucClkZero;      
    u8 ucClkPre = 0x02;       
    u8 ucClkPost;      
    u8 ucHsLpx;        
    u8 ucHsPrep;       
    u8 ucHsTrail;          
    u8 ucHsRqStPre;
    u16 usHss = 0;
    u32 ulrdbyteclk  = 0;
    
    HDMI_WriteI2C_Byte(0xff,0xd4);
    if (b1MipiClockburst)
    {
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | 0x80));
    }
    else
    {
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) & 0x7f));
    }

    ulrdbyteclk = Drv_System_FmClkGet(AD_MLTX_WRITE_CLK);
    ulrdbyteclk = ulrdbyteclk / 1000;
    printf("\nulMipiDataRate: %ld, ulrdbyteclk: %ld",ulMipiDataRate, ulrdbyteclk);
    
    ucHsLpx   = ulrdbyteclk * 6 / 100 + 1; //hs lpx > 50ns
    ucHsPrep  = ulrdbyteclk * 6 / 100; //hs prep : (40ns + 4*UI)~(85ns + 6*UI) , clk_prepare
    ucHsTrail = ulrdbyteclk * 7 / 100 + 4; //hs_trail and clk_trail: max(8UI , 60ns + 4UI),   +2->+3
    ucClkPost = ulrdbyteclk * 7 / 100 + 7; //ck_post > 60ns + 52UI,  +4->+15->+1
    
    if(b1MipiClockburst)
    {
        //������ʱ�Ӵ�1.94G~2.5G�£�clk_zero���ݹ̶���ʽ�޷���ͼ
        if (ulMipiDataRate > CTS_DATARATE)
        {
            ucClkZero = 0x05;
        }
        else
        {
            ucClkZero = ulrdbyteclk * 6 / 25;    //ck_zero > 300 - ck_prpr , old: rdbyteclk/4      
        }
        
        ucHsRqStPre = ucHsLpx + ucHsPrep + ucClkZero + ucClkPre;
    }
    else
    {
        ucClkZero   = ulrdbyteclk * 6 / 25;    //ck_zero > 300 - ck_prpr , old: rdbyteclk/4      
        ucHsRqStPre = ulrdbyteclk / 10;
    }
    
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x8a,ucHsRqStPre);
    HDMI_WriteI2C_Byte(0xa4,ucHsLpx);
    HDMI_WriteI2C_Byte(0xa5,ucHsPrep);
    HDMI_WriteI2C_Byte(0xa6,ucHsTrail);
    HDMI_WriteI2C_Byte(0xa7,ucClkZero);
    HDMI_WriteI2C_Byte(0xa9,ucClkPost);

    printf("\nck_post (0xD4A9) = 0x%02x", ucClkPost);
    printf("\nck_zero (0xD4A7) = 0x%02x", ucClkZero);
    printf("\nhs_lpx  (0xD4A4) = 0x%02x", ucHsLpx);
    printf("\nhs_prep (0xD4A5) = 0x%02x", ucHsPrep);
    printf("\nhs_trail(0xD4A6) = 0x%02x", ucHsTrail);
    printf("\nhs_rqst (0xD48A) = 0x%02x", ucHsRqStPre);

    #if (MIPITX_PATTERN_OUT_SEL == MIPI_DSI)
    usHss = 0x0A;
    #else
    if(b1MipiClockburst == ENABLED)
    {
        usHss = (3 * ucHsRqStPre + ucHsTrail + 9) / 2 + 20;  //usHss
    }
    else
    {
        usHss = (ucHsRqStPre + (ucHsTrail + 13) / 2) + 20;
    }    
    #endif
    
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0x84,(HDMI_ReadI2C_Byte(0x84) & 0xfc));
    HDMI_WriteI2C_Byte(0x84,(u8)((HDMI_ReadI2C_Byte(0x84) | ((usHss >> 8) & 0x03))));
    HDMI_WriteI2C_Byte(0x85,(u8)usHss);
}

void Mod_MipiTx_DPhyClkHsTrig_ForPtn(void)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) & 0xdf));
    usleep(5*1000);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) | BIT5_1)); //[5]:Dphy clk lane hs mode initial trigger
    usleep(1*1000);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) & 0xdf));
}

void Mod_MipiTx_DPhySkewCali_ForPtn(void)
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) | 0x0f)); //RGD_SKEW_CALI_PT_CFG[3:0]
    HDMI_WriteI2C_Byte(0xac,0xff); //RGD_SKEW_CALI_LEN[15:8]
    HDMI_WriteI2C_Byte(0xad,0xf0); //RGD_SKEW_CALI_LEN[7:0]
    HDMI_WriteI2C_Byte(0xae,0x20); //RGD_SKEW_CALI_HS_ZERO[7:0]
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) & BIT4_0));
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) | BIT4_1));
    usleep(5*1000);
    HDMI_WriteI2C_Byte(0xab,(HDMI_ReadI2C_Byte(0xab) & BIT4_0));
    HDMI_WriteI2C_Byte(0xae,0x00);
}

void Mod_MipiTxDig_Set_ForPtn()
{
    HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0xaa,0xAA);   //clk lane data
    HDMI_WriteI2C_Byte(0xab,0x60);
    HDMI_WriteI2C_Byte(0xb6,0x10);
    
    HDMI_WriteI2C_Byte(0x82,0x00);
    HDMI_WriteI2C_Byte(0x83,0xfc);   //read delay
    
    #if TX_VID_PATTERN_SEL == PTN_3840x2160_30
   HDMI_WriteI2C_Byte(0x82,0x00);
    HDMI_WriteI2C_Byte(0x83,0xec);   //read delay
    #elif TX_VID_PATTERN_SEL == PTN_1920x1080_60
    HDMI_WriteI2C_Byte(0x82,0x00);
    HDMI_WriteI2C_Byte(0x83,0xf0);   //read delay
    #elif TX_VID_PATTERN_SEL == PTN_1280x720_60
    HDMI_WriteI2C_Byte(0x82,0x01);
    HDMI_WriteI2C_Byte(0x83,0xec);   //read delay
    #elif TX_VID_PATTERN_SEL == PTN_720x480_60
    HDMI_WriteI2C_Byte(0x82,0x00);
    HDMI_WriteI2C_Byte(0x83,0xec);   //read delay
    #elif TX_VID_PATTERN_SEL == PTN_640x480_60
    HDMI_WriteI2C_Byte(0x82,0x00);
    HDMI_WriteI2C_Byte(0x83,0xf0);   //read delay
    #elif TX_VID_PATTERN_SEL == PTN_640x480_15
    HDMI_WriteI2C_Byte(0x82,0x01);
    HDMI_WriteI2C_Byte(0x83,0x50);   //read delay
    #endif
    
  
    
    HDMI_WriteI2C_Byte(0x87,0x09);   //p1_en
    printf("\nreaddly = 0x%02x%02x",(HDMI_ReadI2C_Byte(0x82)), (HDMI_ReadI2C_Byte(0x83)));
    
    #if (MIPITX_PATTERN_OUT_SEL == MIPI_DSI)
        #if MIPITX_PATTERN_COLORSPACE == RGB_888
        HDMI_WriteI2C_Byte(0x86,0x3e);	 //data type RGB888
        HDMI_WriteI2C_Byte(0x88,0x21);   //24bpp, RGB888
        #elif MIPITX_PATTERN_COLORSPACE == YUV422
        HDMI_WriteI2C_Byte(0x86,0x2c);	 
        HDMI_WriteI2C_Byte(0x88,0x01);   
        #elif MIPITX_PATTERN_COLORSPACE == RGB_666L
        HDMI_WriteI2C_Byte(0x86,0x2E);	 
        HDMI_WriteI2C_Byte(0x88,0x22); 
        #elif MIPITX_PATTERN_COLORSPACE == RGB_666
        HDMI_WriteI2C_Byte(0x86,0x1E);	 
        HDMI_WriteI2C_Byte(0x88,0x10); 
        #elif MIPITX_PATTERN_COLORSPACE == RGB_565
        HDMI_WriteI2C_Byte(0x86,0x0E);	 
        HDMI_WriteI2C_Byte(0x88,0x00); 
        #endif
		
		Drv_MipiTx_PanelInit();
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | BIT4_1));

        if (ucTxLaneNum != MIPITX_4LANE)
        {
            HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | ucTxLaneNum));
        }
        printf("\nMipi DSI Out");
    #else
        
        #if MIPITX_PATTERN_COLORSPACE == RGB_888
        HDMI_WriteI2C_Byte(0x86,0x24); //csi_rgb888
        HDMI_WriteI2C_Byte(0x88,0x25);
        #elif MIPITX_PATTERN_COLORSPACE == YUV422
        HDMI_WriteI2C_Byte(0x86,0x1e); //csi_yuv422_8bit
        HDMI_WriteI2C_Byte(0x88,0x01);
       
        #elif MIPITX_PATTERN_COLORSPACE == RGB_666
        HDMI_WriteI2C_Byte(0x86,0x23); //csi_RGB_666
        HDMI_WriteI2C_Byte(0x88,0x11);
        #elif MIPITX_PATTERN_COLORSPACE == RGB_565
        HDMI_WriteI2C_Byte(0x86,0x22); //csi_RGB_565
        HDMI_WriteI2C_Byte(0x88,0x02);
        #endif
        
        HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | BIT5_1));
        if (ucTxLaneNum != MIPITX_4LANE)
        {
            HDMI_WriteI2C_Byte(0x89,(HDMI_ReadI2C_Byte(0x89) | ucTxLaneNum));
        }
        printf("\nMipi CSI Out");
    #endif
}

void Mod_MipiTxPattern_Out()
{
    printf("\n*************LT9211C MIPITX Pattern Config*************");
    Mod_SystemInt();
    Mod_MipiTx_PtnTiming_Set(&ptn_timing);
    Mod_DesscPll_ForPtn(&ptn_timing);

    /********MIPITX OUTPUT CONFIG********/
    Mod_MipiTxpll_ForPtn(&ptn_timing);
    Mod_MipiTxPhy_ForPtn();
    Mod_MipiTx_TimingSet_ForPtn(&ptn_timing);
    Mod_MipiTx_PhyTimingParaSet_ForPtn();
    Mod_MipiTx_DPhyClkHsTrig_ForPtn();
    Mod_MipiTx_DPhySkewCali_ForPtn();
    Mod_MipiTxDig_Set_ForPtn();
    Mod_VidChkDebug_ForPtn();
    printf("\n*************LT9211C MIPITX Vid Pattern Out************");
    while(1);
}

#endif

#if (TX_PATTERN_SRC_SEL == LVDSTX_PATTERN)


void Mod_LvdsTx_PtnTiming_Set(struct ChipRxVidTiming *ptn_timing)
{

    /* CLK sel */
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x80,0x01); //[7:6]rx sram rd clk src sel ad dessc pcr clk
                                   //[5:4]rx sram wr clk src sel mlrx bytr clk
                                   //[1:0]video check clk sel from desscpll pix clk
    HDMI_WriteI2C_Byte(0x81,0x10); //[5]mlrx byte clock select from ad_mlrxb_byte_clk
                                   //[4]rx output pixel clock select from ad_desscpll_pix_clk
                                   
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x30,0x43); //Chip active RX select pattern
    
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0x25,0x11); //Test video pattern encode mode select
    HDMI_WriteI2C_Byte(0x26,0x04); //

    HDMI_WriteI2C_Byte(0x11,0x77); //pattern mode sel
    HDMI_WriteI2C_Byte(0x12,0xff); //Pattern data value set

	HDMI_WriteI2C_Byte(0x13,(u8)((ptn_timing->usHs+ptn_timing->usHbp)/256));
	HDMI_WriteI2C_Byte(0x14,(u8)((ptn_timing->usHs+ptn_timing->usHbp)%256)); //h_start

	HDMI_WriteI2C_Byte(0x16,(u8)((ptn_timing->usVs+ptn_timing->usVbp)%256)); //v_start

   	HDMI_WriteI2C_Byte(0x17,(u8)(ptn_timing->usHact/256));
	HDMI_WriteI2C_Byte(0x18,(u8)(ptn_timing->usHact%256)); //hactive

	HDMI_WriteI2C_Byte(0x19,(u8)(ptn_timing->usVact/256));
	HDMI_WriteI2C_Byte(0x1a,(u8)(ptn_timing->usVact%256));  //vactive

   	HDMI_WriteI2C_Byte(0x1b,(u8)(ptn_timing->usHtotal/256));
	HDMI_WriteI2C_Byte(0x1c,(u8)(ptn_timing->usHtotal%256)); //htotal

   	HDMI_WriteI2C_Byte(0x1d,(u8)(ptn_timing->usVtotal/256));
	HDMI_WriteI2C_Byte(0x1e,(u8)(ptn_timing->usVtotal%256)); //vtotal

   	HDMI_WriteI2C_Byte(0x1f,(u8)(ptn_timing->usHs/256)); 
	HDMI_WriteI2C_Byte(0x20,(u8)(ptn_timing->usHs%256)); //hs

    HDMI_WriteI2C_Byte(0x21,(u8)(ptn_timing->usVs%256)); //vs

}

void Mod_LvdsTx_DesscPll_ForPtn(struct ChipRxVidTiming *ptn_timing)
{
    u32 pclk_khz;
    u8 desscpll_pixck_div =0;
    u32 pcr_m, pcr_k;

    pclk_khz = ptn_timing->ulPclk_Khz; 
 
    
  	/* dessc pll */
    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x26,0x20); //[7:6]desscpll reference select Xtal clock as reference
                                   //[4]1'b0: dessc-pll power down
	if(pclk_khz < 44000)
	{
        HDMI_WriteI2C_Byte(0x2f,0x07);
		desscpll_pixck_div = 16;
	}

	else if(pclk_khz < 88000)
	{
	  	HDMI_WriteI2C_Byte(0x2f,0x06);
		desscpll_pixck_div = 8;
	}

	else if(pclk_khz < 176000)
	{
	  	HDMI_WriteI2C_Byte(0x2f,0x05);
		desscpll_pixck_div = 4;
	}

	else if(pclk_khz < 352000)
	{
	  	HDMI_WriteI2C_Byte(0x2f,0x04);
		desscpll_pixck_div = 2;
	}

    pcr_m = (pclk_khz * desscpll_pixck_div) /25;
	pcr_k = pcr_m%1000;
	pcr_m = pcr_m/1000;

	pcr_k <<= 14;
	
	//pixel clk
 	HDMI_WriteI2C_Byte(0xff,0xd0); //pcr
	HDMI_WriteI2C_Byte(0x2d,0x7f);
	HDMI_WriteI2C_Byte(0x31,0x00);

	HDMI_WriteI2C_Byte(0x26,0x80|((u8)pcr_m));
	HDMI_WriteI2C_Byte(0x27,(u8)((pcr_k>>16)&0xff)); //K
	HDMI_WriteI2C_Byte(0x28,(u8)((pcr_k>>8)&0xff)); //K
	HDMI_WriteI2C_Byte(0x29,(u8)(pcr_k&0xff)); //K

    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x03,0xfe); //desscpll rst
    HDMI_WriteI2C_Byte(0x03,0xff);
}

void Mod_LvdsTxPhy_ForPtn(void)
{		

#if LVDSTX_PORT_SEL  == PORTA
    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x36,0x01); //lvds enable
    HDMI_WriteI2C_Byte(0x37,0x40);
    #if LVDSTX_LANENUM == FIVE_LANE
        HDMI_WriteI2C_Byte(0x36,0x03); 
    #endif
    
#elif LVDSTX_PORT_SEL  == PORTB
    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x36,0x02); //lvds enable
    HDMI_WriteI2C_Byte(0x37,0x04);
    
#elif LVDSTX_PORT_SEL  == DOU_PORT
    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x36,0x03); //lvds enable
    HDMI_WriteI2C_Byte(0x37,0x44); //port rterm enable
#endif    
    
    HDMI_WriteI2C_Byte(0x38,0x14);
    HDMI_WriteI2C_Byte(0x39,0x31); //0x31
    HDMI_WriteI2C_Byte(0x3a,0xc8);
    HDMI_WriteI2C_Byte(0x3b,0x00);
    HDMI_WriteI2C_Byte(0x3c,0x0f);

    HDMI_WriteI2C_Byte(0x46,0x40);
    HDMI_WriteI2C_Byte(0x47,0x40);
    HDMI_WriteI2C_Byte(0x48,0x40);
    HDMI_WriteI2C_Byte(0x49,0x40);
    HDMI_WriteI2C_Byte(0x4a,0x40);
    HDMI_WriteI2C_Byte(0x4b,0x40);
    HDMI_WriteI2C_Byte(0x4c,0x40);
    HDMI_WriteI2C_Byte(0x4d,0x40);
    HDMI_WriteI2C_Byte(0x4e,0x40);
    HDMI_WriteI2C_Byte(0x4f,0x40);
    HDMI_WriteI2C_Byte(0x50,0x40);
    HDMI_WriteI2C_Byte(0x51,0x40);

    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x03,0xbf); //mltx reset
    HDMI_WriteI2C_Byte(0x03,0xff); //mltx release

}

void Mod_LvdsTxpll_ForPtn(struct ChipRxVidTiming *ptn_timing)
{
    
    u8 ucPreDiv = 0;
    u8 ucSericlkDiv = 0;
    u8 ucDivSet = 0;
    u8 temp = 1;
    float ucPixClkDiv = 0;
    u32 ulLvdsTXPhyClk = 0;
    
    u32 pclk_khz;
    pclk_khz = ptn_timing->ulPclk_Khz; 

    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x30,0x00); //[7]0:txpll normal work; txpll ref clk sel pix clk
    
    /* txphyclk = vco clk * ucSericlkDiv */
#if (LVDSTX_PORT_SEL == DOU_PORT)
    ulLvdsTXPhyClk = (u32)(pclk_khz * 7 / 2); //2 port: byte clk = pix clk / 2;
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | BIT0_1)); //htotal -> 2n
    temp = 2;
    #if ((LVDSTX_COLORSPACE == YUV422) && (LVDSTX_COLORDEPTH == DEPTH_8BIT) && (LVDSTX_LANENUM == FIVE_LANE))
    ulLvdsTXPhyClk = (u32)(pclk_khz * 7 / 4); //2 port YUV422 8bit 5lane: byte clk = pix clk / 4;
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | BIT1_1)); //htotal -> 4n
    temp = 4;
    #endif 
#else
    ulLvdsTXPhyClk = (u32)(pclk_khz * 7); //1 port: byte clk = pix clk;
    temp = 1;
    #if ((LVDSTX_COLORSPACE == YUV422) && (LVDSTX_COLORDEPTH == DEPTH_8BIT) && (LVDSTX_LANENUM == FIVE_LANE))
    ulLvdsTXPhyClk = (u32)(pclk_khz * 7 / 2); //1 port YUV422 8bit 5lane: byte clk = pix clk / 2;
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | BIT0_1)); //htotal -> 2n
    temp = 2;
    #endif
#endif
    
    /*txpll prediv sel*/
    HDMI_WriteI2C_Byte(0xff,0x82);
    if (pclk_khz < 20000)
    {
        HDMI_WriteI2C_Byte(0x31,0x28); //[2:0]3'b000: pre div set div1
        ucPreDiv = 1;
    }
    else if (pclk_khz >= 20000 && pclk_khz < 40000)
    {
        HDMI_WriteI2C_Byte(0x31,0x28); //[2:0]3'b000: pre div set div1
        ucPreDiv = 1;
    }
    else if (pclk_khz >= 40000 && pclk_khz < 80000)
    {
        HDMI_WriteI2C_Byte(0x31,0x29); //[2:0]3'b001: pre div set div2
        ucPreDiv = 2;
    }
    else if (pclk_khz >= 80000 && pclk_khz < 160000)
    {
        HDMI_WriteI2C_Byte(0x31,0x2a); //[2:0]3'b010: pre div set div4
        ucPreDiv = 4;
    }
    else if (pclk_khz >= 160000 && pclk_khz < 320000)
    {
        HDMI_WriteI2C_Byte(0x31,0x2b); //[2:0]3'b011: pre div set div8
        ucPreDiv = 8;
//        ulLvdsTXPhyClk = ulDessc_Pix_Clk * 3.5;
    }
    else if (pclk_khz >= 320000)
    {
        HDMI_WriteI2C_Byte(0x31,0x2f); //[2:0]3'b111: pre div set div16
        ucPreDiv = 16;
//        ulLvdsTXPhyClk = ulDessc_Pix_Clk * 3.5;
    }

    /*txpll serick_divsel*/
    HDMI_WriteI2C_Byte(0xff,0x82);
    if (ulLvdsTXPhyClk >= 640000 )//640M~1.28G
    {
        HDMI_WriteI2C_Byte(0x32,0x42);
        ucSericlkDiv = 1; //sericlk div1 [6:4]:0x40
    }
    else if (ulLvdsTXPhyClk >= 320000 && ulLvdsTXPhyClk < 640000)
    {
        HDMI_WriteI2C_Byte(0x32,0x02);
        ucSericlkDiv = 2; //sericlk div2 [6:4]:0x00
    }
    else if (ulLvdsTXPhyClk >= 160000 && ulLvdsTXPhyClk < 320000)
    {
        HDMI_WriteI2C_Byte(0x32,0x12);
        ucSericlkDiv = 4; //sericlk div4 [6:4]:0x10
    }
    else if (ulLvdsTXPhyClk >= 80000 && ulLvdsTXPhyClk < 160000)
    {
        HDMI_WriteI2C_Byte(0x32,0x22);
        ucSericlkDiv = 8; //sericlk div8 [6:4]:0x20
    }
    else //40M~80M
    {
        HDMI_WriteI2C_Byte(0x32,0x32);
        ucSericlkDiv = 16; //sericlk div16 [6:4]:0x30
    }

    /* txpll_pix_mux_sel & txpll_pixdiv_sel*/
    HDMI_WriteI2C_Byte(0xff,0x82);
    if (pclk_khz > 150000)
    {
        HDMI_WriteI2C_Byte(0x33,0x04); //pixclk > 150000, pixclk mux sel (vco clk / 3.5)
        ucPixClkDiv = 3.5;
    }
    else
    {
       // ucPixClkDiv = (u8)((ulLvdsTXPhyClk * ucSericlkDiv) / (pclk_khz * 7));
         ucPixClkDiv = (u8)(ucSericlkDiv/temp);//modify 2013.11.13
        if (ucPixClkDiv <= 1)
        {
            HDMI_WriteI2C_Byte(0x33,0x00); //pixclk div sel /7
        }
        else if ((ucPixClkDiv > 1) && (ucPixClkDiv <= 2))
        {
            HDMI_WriteI2C_Byte(0x33,0x01); //pixclk div sel /14
        }
        else if ((ucPixClkDiv > 2) && (ucPixClkDiv <= 4))
        {
            HDMI_WriteI2C_Byte(0x33,0x02); //pixclk div sel /28
        }
        else if ((ucPixClkDiv > 4) && (ucPixClkDiv <= 8))
        {
            HDMI_WriteI2C_Byte(0x33,0x03); //pixclk div sel /56
        }
        else
        {
            HDMI_WriteI2C_Byte(0x33,0x03); //pixclk div sel /56
        }
    }
    
   // ucDivSet = (u8)((ulLvdsTXPhyClk * ucSericlkDiv) / (pclk_khz / ucPreDiv));
    ucDivSet = (u8)((7 * ucSericlkDiv*ucPreDiv)/temp);//2023.11.13 modify ucSericlkDiv*ucPreDiv will be not <4
    
	HDMI_WriteI2C_Byte(0x34,0x01); //txpll div set software output enable
    HDMI_WriteI2C_Byte(0x35,ucDivSet);
    printf("\nulPclk_Khz: %ld, ucPreDiv: %d, ucSericlkDiv: %d, ucPixClkDiv: %.1f, ucDivSet: %d",
                    pclk_khz, ucPreDiv, ucSericlkDiv, ucPixClkDiv, ucDivSet);

    printf("\ntemp is: 0x%02x", temp);
    
    #if LVDS_SSC_SEL != NO_SSC

    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x34,0x00); //hardware mode
    
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x6e,0x10); //sram select
    
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x81,0x15); //clk select
    
    HDMI_WriteI2C_Byte(0xff,0x87);

    HDMI_WriteI2C_Byte(0x1e,0x00); //modulation disable
    
    HDMI_WriteI2C_Byte(0x17,0x02); //2 order

    
    HDMI_WriteI2C_Byte(0x18,0x04);
    HDMI_WriteI2C_Byte(0x19,0xd4); //ssc_prd

    #if LVDS_SSC_SEL == SSC_1920x1080_30k2
        HDMI_WriteI2C_Byte(0x1a,0x00);
        HDMI_WriteI2C_Byte(0x1b,0x05); //delta1
        HDMI_WriteI2C_Byte(0x1c,0x00);
        HDMI_WriteI2C_Byte(0x1d,0x0a); //delta
        
        HDMI_WriteI2C_Byte(0x1f,0x1c); //M
        HDMI_WriteI2C_Byte(0x20,0x00);
        HDMI_WriteI2C_Byte(0x21,0x00);
        printf("\nLVDS Output SSC 1920x1080 30k2%");
    #elif LVDS_SSC_SEL == SSC_3840x2160_30k2
        HDMI_WriteI2C_Byte(0x1a,0x00);
        HDMI_WriteI2C_Byte(0x1b,0x05); //delta1
        HDMI_WriteI2C_Byte(0x1c,0x00);
        HDMI_WriteI2C_Byte(0x1d,0x0a); //delta
        
        HDMI_WriteI2C_Byte(0x1f,0x1c); //M
        HDMI_WriteI2C_Byte(0x20,0x00);
        HDMI_WriteI2C_Byte(0x21,0x00);
        printf("\nLVDS Output SSC 3840x2160 30k2%");
    #endif

    HDMI_WriteI2C_Byte(0x1e,0x02); //modulation enable
    
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x0c,0xfe); //txpll reset
    HDMI_WriteI2C_Byte(0x0c,0xff); //txpll release
    
    #endif
}

void Mod_LvdsTxDigital_Forptn(void)
{  
    u16 vss = 0;
    u16 eav = 0;
    u16 sav = 0;
    
    HDMI_WriteI2C_Byte(0xff,0x85); 
#if LVDSTX_MODE == SYNC_MODE
        HDMI_WriteI2C_Byte(0X6e,(HDMI_ReadI2C_Byte(0x6e) & 0xf7));
#elif LVDSTX_MODE == DE_MODE
        HDMI_WriteI2C_Byte(0X6e,(HDMI_ReadI2C_Byte(0x6e) | 0x08)); //[3]lvdstx de mode
#endif        
    
#if ((LVDSTX_PORT_SEL == PORTA) || (LVDSTX_PORT_SEL == PORTB))
        printf("\nLVDS Output Port Num: 1Port");
    	HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x80)); //[7]lvds function enable
    	                                                          //[4]0:output 1port; [4]1:output 2port;
#elif (LVDSTX_PORT_SEL == DOU_PORT)
        printf("\nLVDS Output Port Num: 2Port");
    	HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x90)); //[7]lvds function enable
    	                                                           //[4]0:output 1port; [4]1:output 2port;
#endif    	                               
    	                               
#if (LVDSTX_DATAFORMAT == JEIDA)
        printf("\nData Format: JEIDA");
        HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | BIT6_1)); //[6]1:JEIDA MODE
#elif (LVDSTX_DATAFORMAT == VESA)
        printf("\nData Format: VESA");
        HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) & BIT6_0)); //[6]0:VESA MODE; 
#endif
    
#if (LVDSTX_COLORSPACE == RGB)
        printf("\nColorSpace: RGB");
    #if (LVDSTX_COLORDEPTH == DEPTH_6BIT)
            printf("\nColorDepth: 6Bit");
            HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x40)); //RGB666 [6]RGB666 output must select jeida mode
            HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) & 0xf3));
    #elif (LVDSTX_COLORDEPTH == DEPTH_8BIT)
            printf("\nColorDepth: 8Bit");
            HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x04));
    #elif (LVDSTX_COLORDEPTH == DEPTH_10BIT)
            printf("\nColorDepth: 10Bit");
            HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x0c));
    #endif

#elif (LVDSTX_COLORSPACE == YUV422)

        printf("\nColorSpace: YUV422");
        HDMI_WriteI2C_Byte(0xff,0x86);
        HDMI_WriteI2C_Byte(0x85,0x10); //[4]1: csc RGB to YUV enable
        HDMI_WriteI2C_Byte(0x86,0x40); //[6]1: csc YUV444 to YUV422 enable
        
        HDMI_WriteI2C_Byte(0xff,0x85);
    #if (LVDSTX_COLORDEPTH == DEPTH_8BIT)
            printf("\nColorDepth: 8Bit");
            HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x04));
        #if (LVDSTX_LANENUM == FIVE_LANE)

                printf("\nLvdsLaneNum: 5Lane");
                HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x40)); //YUV422-8bpc-5lane mode output must sel jeida mode
                HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x28));  ////YUV422-8bpc-5lane mode set
        #else
                printf("\nLvdsLaneNum: 4Lane");
                HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) & 0xbf)); //YUV422-8bpc-5lane mode output must sel jeida mode
        #endif
    #endif
#endif

#if (LVDSTX_SYNC_INTER_MODE == ENABLED)
        printf("\nLvds Sync Code Mode: Internal"); //internal sync code mode
    	HDMI_WriteI2C_Byte(0x68,(HDMI_ReadI2C_Byte(0x68) | 0x01));
    #if (LVDSTX_VIDEO_FORMAT == I_FORMAT)
            printf("\nLvds Video Format: interlaced"); //internal sync code mode
            HDMI_WriteI2C_Byte(0xff,0x86);
            HDMI_WriteI2C_Byte(0x11,HDMI_ReadI2C_Byte(0x11) | 0x80); //Pattern Interlace format
            HDMI_WriteI2C_Byte(0xff,0x85);
    	    HDMI_WriteI2C_Byte(0x68,(HDMI_ReadI2C_Byte(0x68) | 0x02));   
    #endif
    #if (LVDSTX_SYNC_CODE_SEND == REPECTIVE)
            printf("\nLvds Sync Code Send: respectively."); //sync code send method sel respectively
    	    HDMI_WriteI2C_Byte(0x68,(HDMI_ReadI2C_Byte(0x68) | 0x04));
    #endif
    
    vss = ptn_timing.usVs + ptn_timing.usVbp;
    eav = ptn_timing.usHs + ptn_timing.usHbp + ptn_timing.usHact + 4;
    sav = ptn_timing.usHs + ptn_timing.usHbp;
    
    
    HDMI_WriteI2C_Byte(0x5f,0x00);    
	HDMI_WriteI2C_Byte(0x60,0x00);  
    HDMI_WriteI2C_Byte(0x62,(u8)(ptn_timing.usVact>>8));         //vact[15:8]
	HDMI_WriteI2C_Byte(0x61,(u8)(ptn_timing.usVact));            //vact[7:0]
    HDMI_WriteI2C_Byte(0x63,(u8)(vss));                         //vss[7:0]
    HDMI_WriteI2C_Byte(0x65,(u8)(eav>>8));                      //eav[15:8]
	HDMI_WriteI2C_Byte(0x64,(u8)(eav));                         //eav[7:0]
    HDMI_WriteI2C_Byte(0x67,(u8)(sav>>8));                      //sav[15:8]
	HDMI_WriteI2C_Byte(0x66,(u8)(sav));                         //sav[7:0]
     
    #if LVDSTX_VIDEO_FORMAT == I_FORMAT
        #if TX_VID_PATTERN_SEL == PTN_1920x1080_30
            HDMI_WriteI2C_Byte(0x61,0x1c);
            HDMI_WriteI2C_Byte(0x62,0x02);
            HDMI_WriteI2C_Byte(0x63,0x14);
        #endif
    #endif

#else
    	HDMI_WriteI2C_Byte(0x68,0x00);
#endif

    HDMI_WriteI2C_Byte(0xff,0x85);
	HDMI_WriteI2C_Byte(0x4a,0x01); //[0]hl_swap_en; [7:6]tx_pt0_src_sel: 0-pta;1-ptb
	HDMI_WriteI2C_Byte(0x4b,0x00);
	HDMI_WriteI2C_Byte(0x4c,0x10);
	HDMI_WriteI2C_Byte(0x4d,0x20);
	HDMI_WriteI2C_Byte(0x4e,0x50);
	HDMI_WriteI2C_Byte(0x4f,0x30);

#if (LVDSTX_LANENUM  == FOUR_LANE)
    	HDMI_WriteI2C_Byte(0x50,0x46); //[7:6]tx_pt1_src_sel: 0-pta;1-ptb
    	HDMI_WriteI2C_Byte(0x51,0x10);
    	HDMI_WriteI2C_Byte(0x52,0x20);
    	HDMI_WriteI2C_Byte(0x53,0x50);
    	HDMI_WriteI2C_Byte(0x54,0x30);
    	HDMI_WriteI2C_Byte(0x55,0x00); //[7:4]pt1_tx4_src_sel
    	HDMI_WriteI2C_Byte(0x56,0x20); //[3:0]pt1_tx5_src_sel
                                       //[6:5]rgd_mltx_src_sel: 0-mipitx;1-lvdstx
   
#elif (LVDSTX_LANENUM == FIVE_LANE)
        HDMI_WriteI2C_Byte(0x50,0x44); //[7:6]tx_pt1_src_sel: 0-pta;1-ptb
    	HDMI_WriteI2C_Byte(0x51,0x00);
    	HDMI_WriteI2C_Byte(0x52,0x10);
    	HDMI_WriteI2C_Byte(0x53,0x20);
    	HDMI_WriteI2C_Byte(0x54,0x50);
    	HDMI_WriteI2C_Byte(0x55,0x30); //[7:4]pt1_tx4_src_sel
    	HDMI_WriteI2C_Byte(0x56,0x24); //[3:0]pt1_tx5_src_sel
                                       //[6:5]rgd_mltx_src_sel: 0-mipitx;1-lvdstx
                                       
        #if 0
            //swap when 9211 lvdstx_to_lvdsrx 2port 5lane link
            HDMI_WriteI2C_Byte(0x54,0x30);
            HDMI_WriteI2C_Byte(0x55,0x40);
        #endif
#endif

    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x08,0x6f); //LVDS TX SW reset
    HDMI_WriteI2C_Byte(0x08,0x7f);

}

void Mod_LvdsTxPort_Swap_Forptn(void)
{
    #if LVDSTX_PORT_SEL == DOU_PORT
        #if LVDSTX_PORT_SWAP == ENABLED
            HDMI_WriteI2C_Byte(0xff,0x85);
            HDMI_WriteI2C_Byte(0x4a,0x41);
            HDMI_WriteI2C_Byte(0x50,(HDMI_ReadI2C_Byte(0x50) & BIT6_0));
            printf("\n2Port Lvds Port Swap");
        #endif
    
    #elif LVDSTX_PORT_SEL == PORTB
        HDMI_WriteI2C_Byte(0xff,0x85);
        HDMI_WriteI2C_Byte(0x4a,0x41);
        HDMI_WriteI2C_Byte(0x50,(HDMI_ReadI2C_Byte(0x50) & BIT6_0));

    #else 
        HDMI_WriteI2C_Byte(0xff,0x85);
        HDMI_WriteI2C_Byte(0x4a,0x01);
        HDMI_WriteI2C_Byte(0x50,(HDMI_ReadI2C_Byte(0x50) | BIT6_1));
    #endif
}

void Mod_LvdsTxPort_Copy_Forptn(void)
{

    #if ((LVDSTX_PORT_COPY == ENABLED) &&  (LVDSTX_PORT_SEL == PORTA))
        HDMI_WriteI2C_Byte(0xff,0x82);
        HDMI_WriteI2C_Byte(0x36,(HDMI_ReadI2C_Byte(0x36) | 0x03)); //port swap enable when porta & portb enable

//        #if (LVDSTX_PORT_SEL == PORTA)
            HDMI_WriteI2C_Byte(0xff,0x85);
            HDMI_WriteI2C_Byte(0x4a,(HDMI_ReadI2C_Byte(0x4a) & 0xbf));
            HDMI_WriteI2C_Byte(0x50,(HDMI_ReadI2C_Byte(0x50) & 0xbf));
            printf("\nPort A Copy");
//        #elif (LVDSTX_PORT_SEL == PORTB)
//            HDMI_WriteI2C_Byte(0xff,0x85);
//            HDMI_WriteI2C_Byte(0x6f,(HDMI_ReadI2C_Byte(0x6f) | 0x10)); //[7]lvds function enable //[4]0:output 1port; [4]1:output 2port;
//            HDMI_WriteI2C_Byte(0x4a,(HDMI_ReadI2C_Byte(0x4a) | 0x40));
//            HDMI_WriteI2C_Byte(0x50,(HDMI_ReadI2C_Byte(0x50) | 0x40));
//            printf("\nPort B Copy");
//        #endif
    
    #endif

}


void Mod_LvdsTxPattern_Out()
{
    printf("\n*************LT9211C LVDSTX Pattern Config*************");
    Mod_SystemInt();
    Mod_LvdsTx_PtnTiming_Set(&ptn_timing);
    Mod_DesscPll_ForPtn(&ptn_timing);
    
    /********LVDSTX OUTPUT CONFIG********/
    Mod_LvdsTxPhy_ForPtn();
	Mod_LvdsTxpll_ForPtn(&ptn_timing);
    Mod_LvdsTxDigital_Forptn();
    Mod_LvdsTxPort_Swap_Forptn();
    Mod_LvdsTxPort_Copy_Forptn();
	Mod_VidChkDebug_ForPtn();
    printf("\n*************LT9211C LVDSTX Vid Pattern Out************");
	while(1);
}

#endif

#if (TX_PATTERN_SRC_SEL == TTLTX_PATTERN)

void Mod_TtlTx_PtnTiming_Set(struct ChipRxVidTiming *ptn_timing)
{

    /* CLK sel */
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x80,0x01); //[7:6]rx sram rd clk src sel ad dessc pcr clk
                                   //[5:4]rx sram wr clk src sel mlrx bytr clk
                                   //[1:0]video check clk sel from desscpll pix clk
    HDMI_WriteI2C_Byte(0x81,0x10); //[5]mlrx byte clock select from ad_mlrxb_byte_clk
                                   //[4]rx output pixel clock select from ad_desscpll_pix_clk
                                   
    HDMI_WriteI2C_Byte(0xff,0x85);
    HDMI_WriteI2C_Byte(0x30,0x43); //Chip active RX select pattern
    
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0x25,0x11); //Test video pattern encode mode select
    HDMI_WriteI2C_Byte(0x26,0x04); //

    HDMI_WriteI2C_Byte(0x11,0x77); //pattern mode sel
    HDMI_WriteI2C_Byte(0x12,0xff); //Pattern data value set

	HDMI_WriteI2C_Byte(0x13,(u8)((ptn_timing->usHs+ptn_timing->usHbp)/256));
	HDMI_WriteI2C_Byte(0x14,(u8)((ptn_timing->usHs+ptn_timing->usHbp)%256)); //h_start

	HDMI_WriteI2C_Byte(0x16,(u8)((ptn_timing->usVs+ptn_timing->usVbp)%256)); //v_start

   	HDMI_WriteI2C_Byte(0x17,(u8)(ptn_timing->usHact/256));
	HDMI_WriteI2C_Byte(0x18,(u8)(ptn_timing->usHact%256)); //hactive

	HDMI_WriteI2C_Byte(0x19,(u8)(ptn_timing->usVact/256));
	HDMI_WriteI2C_Byte(0x1a,(u8)(ptn_timing->usVact%256)); //vactive

   	HDMI_WriteI2C_Byte(0x1b,(u8)(ptn_timing->usHtotal/256));
	HDMI_WriteI2C_Byte(0x1c,(u8)(ptn_timing->usHtotal%256)); //htotal

   	HDMI_WriteI2C_Byte(0x1d,(u8)(ptn_timing->usVtotal/256));
	HDMI_WriteI2C_Byte(0x1e,(u8)(ptn_timing->usVtotal%256)); //vtotal

   	HDMI_WriteI2C_Byte(0x1f,(u8)(ptn_timing->usHs/256)); 
	HDMI_WriteI2C_Byte(0x20,(u8)(ptn_timing->usHs%256)); //hs

    HDMI_WriteI2C_Byte(0x21,(u8)(ptn_timing->usVs%256)); //vs

}

void Mod_TtlTxHalfBit_ForPtn()
{
    #if   ((TTLTX_OUTPUT_MODE == OUTPUT_YCBCR422_8BIT) || (TTLTX_OUTPUT_MODE == OUTPUT_YCBCR422_10BIT) || (TTLTX_OUTPUT_MODE == OUTPUT_YCBCR422_12BIT))
        #define     TTLTX_HALF_BIT_MODE         ENABLED
    #elif ((TTLTX_OUTPUT_MODE == OUTPUT_BT656_8BIT)    || (TTLTX_OUTPUT_MODE == OUTPUT_BT656_10BIT)    || (TTLTX_OUTPUT_MODE == OUTPUT_BT656_12BIT))
        #define     TTLTX_HALF_BIT_MODE         ENABLED
    #elif ((TTLTX_OUTPUT_MODE == OUTPUT_BT1120_8BIT)   || (TTLTX_OUTPUT_MODE == OUTPUT_BT1120_10BIT)   || (TTLTX_OUTPUT_MODE == OUTPUT_BT1120_12BIT))
        #define     TTLTX_HALF_BIT_MODE         ENABLED
    #else
        #define     TTLTX_HALF_BIT_MODE         DISABLED
    #endif
}

void Mod_TtlTxSyncInter_ForPtn()
{
    #if   ((TTLTX_OUTPUT_MODE == OUTPUT_BT656_8BIT)     || (TTLTX_OUTPUT_MODE == OUTPUT_BT656_10BIT)    || (TTLTX_OUTPUT_MODE == OUTPUT_BT656_12BIT))
        #define     TTLTX_SYNC_INTER_MODE       ENABLED
        #define     TTLTX_SYNC_INTER_FORMAT     BT656
    #elif ((TTLTX_OUTPUT_MODE == OUTPUT_BT1120_8BIT)    || (TTLTX_OUTPUT_MODE == OUTPUT_BT1120_10BIT)   || (TTLTX_OUTPUT_MODE == OUTPUT_BT1120_12BIT))
        #define     TTLTX_SYNC_INTER_MODE       ENABLED
        #define     TTLTX_SYNC_INTER_FORMAT     BT1120
    #elif ((TTLTX_OUTPUT_MODE == OUTPUT_BTA_T1004_16BIT)|| (TTLTX_OUTPUT_MODE == OUTPUT_BTA_T1004_20BIT)|| (TTLTX_OUTPUT_MODE == OUTPUT_BTA_T1004_24BIT))
        #define     TTLTX_SYNC_INTER_MODE       ENABLED
        #define     TTLTX_SYNC_INTER_FORMAT     BTA_T1004
    #elif ((TTLTX_OUTPUT_MODE == OUTPUT_BT1120_16BIT)   || (TTLTX_OUTPUT_MODE == OUTPUT_BT1120_20BIT)   || (TTLTX_OUTPUT_MODE == OUTPUT_BT1120_24BIT))
        #define     TTLTX_SYNC_INTER_MODE       ENABLED
        #define     TTLTX_SYNC_INTER_FORMAT     BT1120
    #else
        #define     TTLTX_SYNC_INTER_MODE       DISABLED
    #endif
}

void Mod_TtlTxColorSpace_ForPtn()
{
    #if   ((TTLTX_OUTPUT_MODE == OUTPUT_RGB888) || (TTLTX_OUTPUT_MODE == OUTPUT_RGB666) || (TTLTX_OUTPUT_MODE == OUTPUT_RGB565))
        #define     TTLTX_COLORSPACE            RGB
    #elif TTLTX_OUTPUT_MODE == OUTPUT_YCBCR444
        #define     TTLTX_COLORSPACE            YUV444
    #else
        #define     TTLTX_COLORSPACE            YUV422
    #endif
}

void Mod_TtlTxDdrMode_ForPtn()
{
    #if TTLTX_MODE == SDR
        #define     TTLTX_DATARATE_MODE         SDR
        #define     TTLTX_HALF_CLOCK_MODE       DISABLED
        #define     TTLTX_HALF_WIDTH_MODE       DISABLED
    #elif TTLTX_MODE == DDR
        #define     TTLTX_DATARATE_MODE         DDR
        #define     TTLTX_HALF_CLOCK_MODE       ENABLED
        #define     TTLTX_HALF_WIDTH_MODE       DISABLED
    #elif TTLTX_MODE == DPI
        #define     TTLTX_DATARATE_MODE         DDR
        #define     TTLTX_HALF_CLOCK_MODE       DISABLED
        #define     TTLTX_HALF_WIDTH_MODE       ENABLED
    #endif
}

void Mod_TtlTxDefine_set()
{
    Mod_TtlTxHalfBit_ForPtn();
    Mod_TtlTxSyncInter_ForPtn();
    Mod_TtlTxColorSpace_ForPtn();
    Mod_TtlTxDdrMode_ForPtn();
}

void Mod_TtlTxPhy_ForPtn()
{
    HDMI_WriteI2C_Byte(0xff,0x82);
    HDMI_WriteI2C_Byte(0x64,0xff);
    HDMI_WriteI2C_Byte(0x65,0x80); //dll phase
    
    HDMI_WriteI2C_Byte(0x6D,0xff);
    
    #if TTLTX_DATARATE_MODE == DDR
        #if TTLTX_HALF_CLOCK_MODE == ENABLED
            HDMI_WriteI2C_Byte(0x63,0x0b); //dll reset, ddr on
            HDMI_WriteI2C_Byte(0x63,0x2b); //dll release
            HDMI_WriteI2C_Byte(0x63,0x2f); //phdly_en on
            printf("\nDDR Half Clock Mode:    Enable");
        #endif
        #if TTLTX_HALF_WIDTH_MODE == ENABLED
            //desscpll set
            HDMI_WriteI2C_Byte(0x29,HDMI_ReadI2C_Byte(0x29) | 0x10); //rg_desscpll_phclk_en = 1
            //div set: 822c[5:4] same as 822f[1:0]
            if(HDMI_ReadI2C_Byte(0x2f) & 0x01)
            {
                HDMI_WriteI2C_Byte(0x2c,HDMI_ReadI2C_Byte(0x2c) | 0x10);
            }
            if(HDMI_ReadI2C_Byte(0x2f) & 0x02)
            {
                HDMI_WriteI2C_Byte(0x2c,HDMI_ReadI2C_Byte(0x2c) | 0x20);
            }
            //div set: 822d[3:2] same as 822c[1:0]
            if(HDMI_ReadI2C_Byte(0x2c) & 0x01)
            {
                HDMI_WriteI2C_Byte(0x2d,HDMI_ReadI2C_Byte(0x2d) | 0x04);
            }
            if(HDMI_ReadI2C_Byte(0x2c) & 0x02)
            {
                HDMI_WriteI2C_Byte(0x2d,HDMI_ReadI2C_Byte(0x2d) | 0x08);
            }
            HDMI_WriteI2C_Byte(0x2b,HDMI_ReadI2C_Byte(0x2b) | 0x01); //dessc double pix clock output
            
            HDMI_WriteI2C_Byte(0x63,0x0b); //dll reset, ddr on
            HDMI_WriteI2C_Byte(0x63,0x2b); //dll release
            HDMI_WriteI2C_Byte(0x63,0x2f); //phdly_en on
            printf("\nDDR Half Width Mode:    Enable");
        #endif
    #else
        HDMI_WriteI2C_Byte(0x63,0x17); //dll reset
        HDMI_WriteI2C_Byte(0x63,0x37); //dll release
    #endif
    
    #if TTLTX_HALF_BIT_MODE == ENABLED
        //desscpll set
        HDMI_WriteI2C_Byte(0x29,HDMI_ReadI2C_Byte(0x29) | 0x10); //rg_desscpll_phclk_en = 1
        //div set: 822c[5:4] same as 822f[1:0]
        if(HDMI_ReadI2C_Byte(0x2f) & 0x01)
        {
            HDMI_WriteI2C_Byte(0x2c,HDMI_ReadI2C_Byte(0x2c) | 0x10);
        }
        if(HDMI_ReadI2C_Byte(0x2f) & 0x02)
        {
            HDMI_WriteI2C_Byte(0x2c,HDMI_ReadI2C_Byte(0x2c) | 0x20);
        }
        //div set: 822d[3:2] same as 822c[1:0]
        if(HDMI_ReadI2C_Byte(0x2c) & 0x01)
        {
            HDMI_WriteI2C_Byte(0x2d,HDMI_ReadI2C_Byte(0x2d) | 0x04);
        }
        if(HDMI_ReadI2C_Byte(0x2c) & 0x02)
        {
            HDMI_WriteI2C_Byte(0x2d,HDMI_ReadI2C_Byte(0x2d) | 0x08);
        }
        HDMI_WriteI2C_Byte(0x2b,HDMI_ReadI2C_Byte(0x2b) | 0x01); //dessc double pix clock output
        printf("\nHalf Bit Mode:    Enable");
    #endif
}

void Mod_TtlTxDig_ForPtn()
{
    HDMI_WriteI2C_Byte(0xff,0x86);
    HDMI_WriteI2C_Byte(0xa5,0x00);
    HDMI_WriteI2C_Byte(0xa8,0x00);
    HDMI_WriteI2C_Byte(0xa9,0x00);
    HDMI_WriteI2C_Byte(0xaf,0x3f);
    

    #if TTLTX_SYNC_INTER_MODE == ENABLED
        #if TTLTX_HALF_BIT_MODE == ENABLED
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) | 0x04); //8bit embedded enable
            printf("\nSync Mode:    8Bit Embedded");
        #else
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) | 0x02); //16bit embedded enable
            printf("\nSync Mode:    16Bit Embedded");
        #endif
        #if TTLTX_SYNC_INTER_FORMAT == BT1120
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) | 0x01); //BT1120 enable
            printf("\nEmbedded Format:    BT1120");
        #elif TTLTX_SYNC_INTER_FORMAT == BTA_T1004
            printf("\nEmbedded Format:    BTA_T1004");
        #elif TTLTX_SYNC_INTER_FORMAT == BT656
            printf("\nEmbedded Format:    BT656");
        #endif
        #if TTLTX_VSYNC_POLARITY == PLUS
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) | 0x20); //vsync polarity
            printf("\nVsync Polarity:    plus");
        #else
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) & 0xdf); //vsync polarity
            printf("\nVsync Polarity:    minus");
        #endif
        #if TTLTX_HSYNC_POLARITY == PLUS
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) | 0x10); //hsync polarity
            printf("\nHsync Polarity:    plus");
        #else
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) & 0xef); //hsync polarity
            printf("\nHsync Polarity:    minus");
        #endif
        #if TTLTX_VIDEO_FORMAT == I_FORMAT
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) | 0x08); //Interlace format
            HDMI_WriteI2C_Byte(0x11,HDMI_ReadI2C_Byte(0x11) | 0x80); //Pattern Interlace format
            HDMI_WriteI2C_Byte(0xa6,HDMI_ReadI2C_Byte(0xa6) | 0x08); //rgd_ttltx_eav_chg_en
            printf("\nVideo Format:    I Format");
        #else
            HDMI_WriteI2C_Byte(0xa5,HDMI_ReadI2C_Byte(0xa5) & 0xf7); //Progressive format
            HDMI_WriteI2C_Byte(0x11,HDMI_ReadI2C_Byte(0x11) & 0x7f); //Pattern Progressive format
            printf("\nVideo Format:    P Format");
        #endif
    #endif

    
    #if TTLTX_DATARATE_MODE == DDR
        #if TTLTX_HALF_WIDTH_MODE == ENABLED
            HDMI_WriteI2C_Byte(0xaf,HDMI_ReadI2C_Byte(0xaf) | 0x80); //DDR Half Width Mode Enable
            #if 0
            HDMI_WriteI2C_Byte(0xaf,HDMI_ReadI2C_Byte(0xaf) | 0x40); //high 12 Bit first
            printf("\nDDR Trans Mode:    High 12Bit First");
            #else
            HDMI_WriteI2C_Byte(0xaf,HDMI_ReadI2C_Byte(0xaf) & 0xbf); //low 12 Bit first
            printf("\nDDR Trans Mode:    Low 12Bit First");
            #endif
        #endif
    #endif
    
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x0d,0x7d); //ttl tx reset
    HDMI_WriteI2C_Byte(0x0d,0x7f); //ttl tx release
}

void Mod_TtlTxMapping_ForPtn()
{
    HDMI_WriteI2C_Byte(0xff,0x86);
    
    switch(TTLTX_OUTPUT_MODE)
    {
        case OUTPUT_RGB888:
            //RGB swap case
            #if 1   //output RGB  R: D23->D16    G: D15->D8    B:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x00);
            #endif
            #if 0   //output RBG  R: D23->D16    B: D15->D8    G:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x30);
            #endif
            #if 0   //output GRB  G: D23->D16    R: D15->D8    B:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            #if 0   //output GBR  G: D23->D16    B: D15->D8    R:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x50);
            #endif
            #if 0   //output BRG  B: D23->D16    R: D15->D8    G:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x60);
            #endif
            #if 0   //output BGR  B: D23->D16    G: D15->D8    R:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            //RGB high/low swap case
            #if 0   //output BGR  B: D23->D16    G: D15->D8    R:D0->D7    
                HDMI_WriteI2C_Byte(0xa8,0x71);
            #endif
            #if 0   //output BGR  B: D23->D16    G: D8->D15    R:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x72);
            #endif
            #if 0   //output BGR  B: D16->D23    G: D15->D8    R:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x74);
            #endif
            printf("\nTTLTx Output Mode:    RGB888");
            break;
            
        case OUTPUT_RGB666:
            //RGB swap case
            #if 1   //output RGB  R: D23->D18    G: D15->D10    B:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x00);
            #endif
            #if 0   //output RBG  R: D23->D18    B: D15->D10    G:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x30);
            #endif
            #if 0   //output GRB  G: D23->D18    R: D15->D10    B:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            #if 0   //output GBR  G: D23->D18    B: D15->D10    R:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x50);
            #endif
            #if 0   //output BRG  B: D23->D18    R: D15->D10    G:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x60);
            #endif
            #if 0   //output BGR  B: D23->D18    G: D15->D10    R:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            //RGB high/low swap case
            #if 0   //output BGR  B: D23->D18    G: D15->D10    R:D0->D5    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x71);
            #endif
            #if 0   //output BGR  B: D23->D18    G: D8->D13    R:D7->D2     
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x72);
            #endif
            #if 0   //output BGR  B: D16->D21    G: D15->D10    R:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x74);
            #endif
            //RGB666 swap case
            #if 0   //output BGR  B: D17->D12    G: D11->D6    R:D5->D0     
                HDMI_WriteI2C_Byte(0xa7,0x05);
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            #if 0   //output BGR  B: D23->D18    G: D17->D12    R:D11->D6   
                HDMI_WriteI2C_Byte(0xa7,0x04);
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            printf("\nTTLTx Output Mode:    RGB666");
            break;
            
        case OUTPUT_RGB565:
            //RGB swap case
            #if 0   //output RGB  R: D23->D19    G: D15->D10    B:D7->D3    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x00);
            #endif
            #if 0   //output RBG  R: D23->D19    B: D15->D11    G:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x30);
            #endif
            #if 0   //output GRB  G: D23->D18    R: D15->D11    B:D7->D3    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            #if 0   //output GBR  G: D23->D18    B: D15->D11    R:D7->D3    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x50);
            #endif
            #if 0   //output BRG  B: D23->D19    R: D15->D11    G:D7->D2    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x60);
            #endif
            #if 0   //output BGR  B: D23->D19    G: D15->D10    R:D7->D3    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            //RGB high/low swap case
            #if 0   //output BGR  B: D23->D19    G: D15->D10    R:D0->D4    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x71);
            #endif
            #if 0   //output BGR  B: D23->D19    G: D8->D13    R:D7->D3   
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x72);
            #endif
            #if 0   //output BGR  B: D16->D20    G: D15->D10    R:D7->D3  
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x74);
            #endif
            //RGB565 case
            #if 1   //output RGB  B: D15->D11    G: D10->D5    R: D4->D0  
                HDMI_WriteI2C_Byte(0xa7,0x07);
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            #if 0   //output RGB  B: D23->D19    G: D18->D13    R: D12->D8
                HDMI_WriteI2C_Byte(0xa7,0x06);
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            printf("\nTTLTx Output Mode:    RGB565");
            break;
        
        case OUTPUT_YCBCR444:
            //RGB swap case
            #if 1   //output YUV  V: D23->D16    Y: D15->D8    U:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x00);
            #endif
            #if 0   //output YUV  V: D23->D16    U: D15->D8    Y:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x30);
            #endif
            #if 0   //output YUV  Y: D23->D16    V: D15->D8    U:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            #if 0   //output YUV  Y: D23->D16    U: D15->D8    V:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x50);
            #endif
            #if 0   //output YUV  U: D23->D16    V: D15->D8    Y:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x60);
            #endif
            #if 0   //output YUV  U: D23->D16    Y: D15->D8    V:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            //RGB high/low swap case
            #if 0   //output YUV  U: D23->D16    Y: D15->D8    V:D0->D7   
                HDMI_WriteI2C_Byte(0xa8,0x71);
            #endif
            #if 0   //output YUV  U: D23->D16    Y: D8->D15    V:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x72);
            #endif
            #if 0   //output YUV  U: D16->D23    Y: D15->D8    V:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x74);
            #endif
            printf("\nTTLTx Output Mode:    YCBCR444");
            break;
        
        case OUTPUT_YCBCR422_8BIT:
            printf("\nTTLTx Output Mode:    YCBCR422_8BIT");
            break;
        
        case OUTPUT_YCBCR422_10BIT:
            printf("\nTTLTx Output Mode:    YCBCR422_10BIT");
            break;
        
        case OUTPUT_YCBCR422_12BIT:
            printf("\nTTLTx Output Mode:    YCBCR422_12BIT");
            break;
        
        case OUTPUT_YCBCR422_16BIT:
            //RGB swap case
            #if 1   //output YUV  C: D23->D16    Y: D15->D8 
                HDMI_WriteI2C_Byte(0xa8,0x00);
            #endif
            #if 0   //output YUV  C: D23->D16    Y:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x30);
            #endif
            #if 0   //output YUV  Y: D23->D16    C: D15->D8 
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            #if 0   //output YUV  Y: D23->D16    C:D7->D0   
                HDMI_WriteI2C_Byte(0xa8,0x50);
            #endif
            #if 0   //output YUV  C: D15->D8    Y:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x60);
            #endif
            #if 0   //output YUV  Y: D15->D8    C:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            //RGB high/low swap case
            #if 0   //output YUV  Y: D15->D8    C:D0->D7    
                HDMI_WriteI2C_Byte(0xa8,0x71);
            #endif
            #if 0   //output YUV  Y: D8->D15    C:D7->D0    
                HDMI_WriteI2C_Byte(0xa8,0x72);
            #endif
            printf("\nTTLTx Output Mode:    YCBCR422_16BIT");
            break;
        
        case OUTPUT_YCBCR422_20BIT:
            //20Bit swap case
            #if 1  //output YUV  Y: D23->D14    C:D13->D4    
                HDMI_WriteI2C_Byte(0xa7,0x02);
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            #if 0  //output YUV  Y: D19->D10    C:D9->D0    
                HDMI_WriteI2C_Byte(0xa7,0x03);
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            //RGB swap case
            #if 0  //output YUV  Y: D15->D8  D19->D18    C:D8->D0  D23->D22    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            #if 0  //output YUV  Y: D23->D16  D3->D2    C:D15->D6   
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            printf("\nTTLTx Output Mode:    YCBCR422_20BIT");
            break;
        
        case OUTPUT_YCBCR422_24BIT:
            //24Bit swap case
            #if 1  //output YUV  Y: D23->D12    C:D11->D0   
                HDMI_WriteI2C_Byte(0xa7,0x01);
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            #if 0  //output YUV  C: D23->D12    Y:D11->D0  
                HDMI_WriteI2C_Byte(0xa7,0x01);
                HDMI_WriteI2C_Byte(0xa8,0x00);
            #endif
            //RGB swap case
            #if 0  //output YUV  Y: D15->D8  D19->D16    C:D7->D0  D23->D20    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x70);
            #endif
            #if 0  //output YUV  Y: D23->D16  D3->D0    C:D15->D4    
                HDMI_WriteI2C_Byte(0xa7,0x00);
                HDMI_WriteI2C_Byte(0xa8,0x40);
            #endif
            printf("\nTTLTx Output Mode:    YCBCR422_24BIT");
            break;
        
        case OUTPUT_BT656_8BIT:
            //RGB swap case
            #if 0  //output YUV  D7-D0   
                HDMI_WriteI2C_Byte(0xa9,0x07);
            #endif
            #if 1  //output YUV  D15-D8    
                HDMI_WriteI2C_Byte(0xa9,0x09);
            #endif
            #if 0  //output YUV  D23-D16   
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            printf("\nTTLTx Output Mode:    BT656_8BIT");
            break;
        
        case OUTPUT_BT656_10BIT:
            //YC swap, 20Bit swap case
            #if 0  //output YUV  D9-D0    
                HDMI_WriteI2C_Byte(0xa9,0x05);
            #endif
            #if 0  //output YUV  D13-D4    
                HDMI_WriteI2C_Byte(0xa9,0x04);
            #endif
            #if 0  //output YUV  D19-D10    
                HDMI_WriteI2C_Byte(0xa9,0x45);
            #endif
            #if 0  //output YUV  D23-D14    
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            //other case
            #if 1  //output YUV  D15-D6    
                HDMI_WriteI2C_Byte(0xa9,0x08);
            #endif
            #if 0  //output YUV  D11-D2 
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            printf("\nTTLTx Output Mode:    BT656_10BIT");
            break;
        
        case OUTPUT_BT656_12BIT:
            //YC swap case
            #if 0  //output YUV  D11-D0   
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            #if 0  //output YUV  D23-D12  
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            //other case
            #if 1  //output YUV  D15-D4    
                HDMI_WriteI2C_Byte(0xa9,0x01);
            #endif
            printf("\nTTLTx Output Mode:    BT656_12BIT");
            break;
        
        case OUTPUT_BT1120_8BIT:
            //RGB swap case
            #if 0  //output YUV  D7-D0   
                HDMI_WriteI2C_Byte(0xa9,0x07);
            #endif
            #if 1  //output YUV  D15-D8    
                HDMI_WriteI2C_Byte(0xa9,0x09);
            #endif
            #if 0  //output YUV  D23-D16   
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            printf("\nTTLTx Output Mode:    BT1120_8BIT");
            break;
        
        case OUTPUT_BT1120_10BIT:
            //YC swap, 20Bit swap case
            #if 0  //output YUV  D9-D0    
                HDMI_WriteI2C_Byte(0xa9,0x05);
            #endif
            #if 0  //output YUV  D13-D4    
                HDMI_WriteI2C_Byte(0xa9,0x04);
            #endif
            #if 0  //output YUV  D19-D10    
                HDMI_WriteI2C_Byte(0xa9,0x45);
            #endif
            #if 0  //output YUV  D23-D14    
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            //other case
            #if 1  //output YUV  D15-D6    
                HDMI_WriteI2C_Byte(0xa9,0x08);
            #endif
            #if 0  //output YUV  D11-D2 
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            printf("\nTTLTx Output Mode:    BT1120_10BIT");
            break;
        
        case OUTPUT_BT1120_12BIT:
            //YC swap case
            #if 0  //output YUV  D11-D0   
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            #if 0  //output YUV  D23-D12  
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            //other case
            #if 1  //output YUV  D15-D4    
                HDMI_WriteI2C_Byte(0xa9,0x01);
            #endif
            printf("\nTTLTx Output Mode:    BT1120_12BIT");
            break;
        
        case OUTPUT_BTA_T1004_16BIT:
            //RGB swap case
            #if 0  //output YUV  C: D7-D0    Y: D15-D8    
                HDMI_WriteI2C_Byte(0xa9,0x47);
            #endif
            #if 0  //output YUV  Y: D7-D0    C: D15-D8    
                HDMI_WriteI2C_Byte(0xa9,0x07);
            #endif
            #if 0  //output YUV  Y: D23-D16    C: D15-D8  
                HDMI_WriteI2C_Byte(0xa9,0x46);
            #endif
            #if 1  //output YUV  C: D23-D16    Y: D15-D8  
                HDMI_WriteI2C_Byte(0xa9,0x06);
            #endif
            //other case
            #if 0  //output YUV  C: D23-D16    Y: D11-D4    
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            printf("\nTTLTx Output Mode:    BTA_T1004_16BIT");
            break;
        
        case OUTPUT_BTA_T1004_20BIT:
            //YC swap, 20Bit swap case
            #if 0  //output YUV  C: D9-D0    Y: D19-D10 
                HDMI_WriteI2C_Byte(0xa9,0x45);
            #endif
            #if 0  //output YUV  Y: D9-D0    C: D19-D10 
                HDMI_WriteI2C_Byte(0xa9,0x05);
            #endif
            #if 0  //output YUV  Y: D23-D14    C: D13-D4
                HDMI_WriteI2C_Byte(0xa9,0x44);
            #endif
            #if 0  //output YUV  C: D23-D14    Y: D13-D4
                HDMI_WriteI2C_Byte(0xa9,0x04);
            #endif
            //other case
            #if 1 //output YUV  C: D23-D14    Y: D11-D2    
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            #if 0 //output YUV  Y: D23-D14    C: D11-D2    
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            printf("\nTTLTx Output Mode:    BTA_T1004_20BIT");
            break;
        
        case OUTPUT_BTA_T1004_24BIT:
            //YC swap case 
            #if 0 //output YUV  Y: D23-D12    C: D11-D0   
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            #if 1 //output YUV  C: D23-D12    Y: D11-D0   
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            printf("\nTTLTx Output Mode:    BTA_T1004_24BIT");
            break;
        
        case OUTPUT_BT1120_16BIT: 
            //RGB swap case
            #if 0  //output YUV  C: D7-D0    Y: D15-D8    
                HDMI_WriteI2C_Byte(0xa9,0x47);
            #endif
            #if 0  //output YUV  Y: D7-D0    C: D15-D8    
                HDMI_WriteI2C_Byte(0xa9,0x07);
            #endif
            #if 0  //output YUV  Y: D23-D16    C: D15-D8  
                HDMI_WriteI2C_Byte(0xa9,0x46);
            #endif
            #if 1  //output YUV  C: D23-D16    Y: D15-D8  
                HDMI_WriteI2C_Byte(0xa9,0x06);
            #endif
            //other case
            #if 0  //output YUV  C: D23-D16    Y: D11-D4    
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            printf("\nTTLTx Output Mode:    BT1120_16BIT");
            break;
        
        case OUTPUT_BT1120_20BIT:
            //YC swap, 20Bit swap case
            #if 0  //output YUV  C: D9-D0    Y: D19-D10 
                HDMI_WriteI2C_Byte(0xa9,0x45);
            #endif
            #if 0  //output YUV  Y: D9-D0    C: D19-D10 
                HDMI_WriteI2C_Byte(0xa9,0x05);
            #endif
            #if 0  //output YUV  Y: D23-D14    C: D13-D4
                HDMI_WriteI2C_Byte(0xa9,0x44);
            #endif
            #if 0  //output YUV  C: D23-D14    Y: D13-D4
                HDMI_WriteI2C_Byte(0xa9,0x04);
            #endif
            //other case
            #if 1 //output YUV  C: D23-D14    Y: D11-D2    
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            #if 0 //output YUV  Y: D23-D14    C: D11-D2    
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            printf("\nTTLTx Output Mode:    BT1120_20BIT");
            break;
        
        case OUTPUT_BT1120_24BIT:
            //YC swap case 
            #if 0 //output YUV  Y: D23-D12    C: D11-D0   
                HDMI_WriteI2C_Byte(0xa9,0x40);
            #endif
            #if 1 //output YUV  C: D23-D12    Y: D11-D0   
                HDMI_WriteI2C_Byte(0xa9,0x00);
            #endif
            printf("\nTTLTx Output Mode:    BT1120_24BIT");
            break;
    }
}

void Mod_TtlTx_BtTiming_Set()
{
    u16 vss,hss;
    
    vss = ptn_timing.usVs + ptn_timing.usVbp;
    hss = ptn_timing.usHs + ptn_timing.usHbp;
    
    HDMI_WriteI2C_Byte(0xff,0x86);
    
    HDMI_WriteI2C_Byte(0xb6,(u8)(ptn_timing.usHtotal>>8));
	HDMI_WriteI2C_Byte(0xb7,(u8)(ptn_timing.usHtotal));  
    HDMI_WriteI2C_Byte(0xb8,(u8)(ptn_timing.usHact>>8));
	HDMI_WriteI2C_Byte(0xb9,(u8)(ptn_timing.usHact)); 
    HDMI_WriteI2C_Byte(0xba,(u8)(hss>>8));
	HDMI_WriteI2C_Byte(0xbb,(u8)(hss)); 
    HDMI_WriteI2C_Byte(0xbc,(u8)(ptn_timing.usVact>>8));
	HDMI_WriteI2C_Byte(0xbd,(u8)(ptn_timing.usVact)); 
    HDMI_WriteI2C_Byte(0xbe,(u8)(vss>>8));
	HDMI_WriteI2C_Byte(0xbf,(u8)(vss));
    
    #if TTLTX_VIDEO_FORMAT == I_FORMAT
        #if TX_VID_PATTERN_SEL == PTN_1920x1080_30
            HDMI_WriteI2C_Byte(0xb1,0x17);  //vs_2nd_blk 
            HDMI_WriteI2C_Byte(0xb3,0x01);  //1st_vcnt
            HDMI_WriteI2C_Byte(0xb4,0x02);  //2nd_vcnt_h
            HDMI_WriteI2C_Byte(0xb5,0x33);  //2nd_vcnt_l
            HDMI_WriteI2C_Byte(0xbf,0x14);
        #endif
    #endif
}

void Mod_CscSet_ForTtlPtn()
{
    #if TTLTX_COLORSPACE == RGB
        printf("\nColor Format:    RGB");
    #elif TTLTX_COLORSPACE == YUV444
        HDMI_WriteI2C_Byte(0xff,0x86);
        HDMI_WriteI2C_Byte(0x85,0x10);
        printf("\nColor Format:    YUV444");
    #elif TTLTX_COLORSPACE == YUV422
        HDMI_WriteI2C_Byte(0xff,0x86);
        HDMI_WriteI2C_Byte(0x85,0x10); //[4]1'b1: rgb to yuv enable
        HDMI_WriteI2C_Byte(0x86,0x40); //[6]1'b1: yuv444 to yuv 422 enable
        printf("\nColor Format:    YUV422");
    #endif
}

//Mod_TtlTxPhaseSet_ForPtn����������ttltx IOʱ�ӵ���λ
void Mod_TtlTxPhaseSet_ForPtn()
{
    //pclk_khz: ttltx�ο���pix_clk
    //io_dclk_khz: ttltxʵ��ͨ��IO�����pin�ŵ�dclk
    //ucphase��DLL ��λ ��Χ0~7
    u32 pclk_khz;
    u32 io_dclk_khz;
    u8 ucphase;
    
    //�ο�ʱ��pix_clkѡ��: 
        //ad_desscpll_pix_clk��patternȫ��ģʽ/mipirxȫ��ģʽ/lvdsrx��desscģʽ��   
        //ad_rxpll_pix_clk��ttlrxȫ��ģʽ/lvdsrx�ķ�desscģʽ��
    pclk_khz = Drv_System_FmClkGet(AD_DESSCPLL_PIX_CLK);
    
    //ttltx��ͬģʽ��io_dclk_khz��pclk_khz�Ķ�Ӧ��ϵ
    #if TTLTX_DATARATE_MODE == SDR// SDR case
        #if TTLTX_HALF_BIT_MODE == DISABLED     //case 1��RGB 24BIT��YUV422 16BIT��BT1120 16BIT 
            io_dclk_khz = pclk_khz;
        #elif TTLTX_HALF_BIT_MODE == ENABLED    //case 2��BT656 8BIT
            io_dclk_khz = pclk_khz*2;
        #endif
    #elif TTLTX_DATARATE_MODE == DDR// DDR case
        #if TTLTX_HALF_CLOCK_MODE == ENABLED//һ��DDRģʽ
            #if TTLTX_HALF_BIT_MODE == DISABLED     //case 1��RGB 24BIT��YUV422 16BIT��BT1120 16BIT
                io_dclk_khz = pclk_khz/2;
            #elif TTLTX_HALF_BIT_MODE == ENABLED    //case 2��BT656 8BIT
                io_dclk_khz = pclk_khz;
            #endif
        #endif
        #if TTLTX_HALF_WIDTH_MODE == ENABLED//DPIģʽ
            io_dclk_khz = pclk_khz;     //RGB 24BIT DPI
        #endif
    #endif
    
    //����io_dclk_khz������λ
    //��λ���÷�����
        //SDRģʽ��DLL ��λĬ��0��ʱ���ź������ز�������Ƶ��io_dclk_khz < 100MHz�����򣬸�Ƶ����������ͻ��Ľ��ն�����ȷҪ�󣬰�Ҫ�����DLL��λ
        //DDRģʽ��Ĭ��DE���һ����Ч���������أ�DLL����λ3��Ĭ��DE���һ����Ч�����½���ʱ��DLL����λ7������ͻ��Ľ��ն�����ȷҪ�󣬰�Ҫ�����DLL��λ
    #if TTLTX_DATARATE_MODE == SDR
        ucphase = 0; //Ĭ����λ0
        HDMI_WriteI2C_Byte(0xff,0x82);
        HDMI_WriteI2C_Byte(0x65,HDMI_ReadI2C_Byte(0x65) | (ucphase<<4)); //DLL ��λ���� ��ע: ��λ����ʧ�ܿɼ��8263[2]�Ƿ�򿪣�
        if(pclk_khz <= 100000)
        {
            HDMI_WriteI2C_Byte(0x2b,HDMI_ReadI2C_Byte(0x2b) | BIT3_1);  //822b[3]=1,ʱ�ӷ���
        }
        else
        {
            HDMI_WriteI2C_Byte(0xff,0x82);
            HDMI_WriteI2C_Byte(0x2b,HDMI_ReadI2C_Byte(0x2b) & BIT3_0);  //822b[3]=1,ʱ�Ӳ�����
        }
    #elif TTLTX_DATARATE_MODE == DDR
        ucphase = 3; //Ĭ����λ3��7
        HDMI_WriteI2C_Byte(0xff,0x82);
        HDMI_WriteI2C_Byte(0x65,HDMI_ReadI2C_Byte(0x65) | (ucphase<<4)); //DLL ��λ���� ��ע: ��λ����ʧ�ܿɼ��8263[2]�Ƿ�򿪣�
    #endif
        
    printf("\npix clk = %ld kHz",pclk_khz);
    printf("\nout clk = %ld kHz",io_dclk_khz);
    printf("\nDLL phase set: 0x%02x", ucphase);
    
}

void Mod_TtlTxPattern_Out()
{
    printf("\n*************LT9211C TTLTX Pattern Config*************");
    Mod_SystemInt();
    Mod_TtlTx_PtnTiming_Set(&ptn_timing);
    Mod_DesscPll_ForPtn(&ptn_timing);

    /********TTLTX OUTPUT CONFIG********/
    Mod_TtlTxDefine_set();
    Mod_TtlTxPhy_ForPtn();
    Mod_TtlTxDig_ForPtn();
    Mod_TtlTxMapping_ForPtn();
    Mod_CscSet_ForTtlPtn();
    #if TTLTX_SYNC_INTER_MODE == ENABLED
        Mod_TtlTx_BtTiming_Set();
    #endif
    Mod_TtlTxPhaseSet_ForPtn();
    
	Mod_VidChkDebug_ForPtn();
    HDMI_WriteI2C_Byte(0xff,0x81);
    HDMI_WriteI2C_Byte(0x0d,0x7d); //ttl tx reset
    HDMI_WriteI2C_Byte(0x0d,0x7f); //ttl tx release
    printf("\n*************LT9211C TTLTX Vid Pattern Out************");
	while(1);
}

#endif

void Mod_ChipTx_PtnOut()
{
    #if TX_PATTERN_SRC_SEL == MIPITX_PATTERN
        Mod_MipiTxPattern_Out();
    #endif
    #if TX_PATTERN_SRC_SEL == LVDSTX_PATTERN
        Mod_LvdsTxPattern_Out();
    #endif
    #if TX_PATTERN_SRC_SEL == TTLTX_PATTERN
        Mod_TtlTxPattern_Out();
    #endif
}

#endif