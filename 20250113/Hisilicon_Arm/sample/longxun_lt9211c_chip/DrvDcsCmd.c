#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
  * @project: LT9211C
  * @file: DrvDcsCmd.c
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#include "include.h"


#if ((LT9211C_MODE_SEL == LVDS_IN_MIPI_OUT)||(LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)||(LT9211C_MODE_SEL == TTL_IN_MIPI_OUT) || (TX_PATTERN_SRC_SEL == MIPITX_PATTERN))


void Drv_MipiTx_DcsPktWrite(u8 DCS_DI, u8 ucLen, u8* Ptr )
{
	u8 i = 0;
	
	HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0xa0,0x00);
	
	if(ucLen == 2)
	{  	
	    if (DCS_DI == 0x29)
	    {
            HDMI_WriteI2C_Byte( 0x9a, 0xe1 );
    		HDMI_WriteI2C_Byte( 0x99, ucLen + 6 );
    		HDMI_WriteI2C_Byte( 0x98, DCS_DI );
    		HDMI_WriteI2C_Byte( 0x98, ucLen );
    		HDMI_WriteI2C_Byte( 0x98, 0x00 );
    	
    		for(i = 0; i < ucLen; i++)
    		{
    			HDMI_WriteI2C_Byte( 0x98, *Ptr );
    			Ptr++;														   
    		}
	    }
	    else
	    {
    		HDMI_WriteI2C_Byte( 0x9a, 0xc1 );
    		HDMI_WriteI2C_Byte( 0x99, 0x04 );
    		HDMI_WriteI2C_Byte( 0x98, DCS_DI );
    		HDMI_WriteI2C_Byte( 0x98, *Ptr );
    		HDMI_WriteI2C_Byte( 0x98, *( Ptr + 1 ) );
		}
	}
	else
	{
		HDMI_WriteI2C_Byte( 0x9a, 0xe1 );
		HDMI_WriteI2C_Byte( 0x99, ucLen + 6 );
		HDMI_WriteI2C_Byte( 0x98, DCS_DI );
		HDMI_WriteI2C_Byte( 0x98, ucLen );
		HDMI_WriteI2C_Byte( 0x98, 0x00 );
	
		for(i = 0; i < ucLen; i++)
		{
			HDMI_WriteI2C_Byte( 0x98, *Ptr );
			Ptr++;														   
		}
	}
	usleep(1*1000);

	HDMI_WriteI2C_Byte( 0xab, 0x00 );

}

void Drv_MipiTx_DcsPktRead(u8 DCS_DI, u8 ucLen, u8* Ptr )
{
	u8 i = 0;
	
	HDMI_WriteI2C_Byte(0xff,0xd4);
    HDMI_WriteI2C_Byte(0xa0,0x10);
	
	if(ucLen == 2)
	{  	
		HDMI_WriteI2C_Byte( 0x9a, 0xc2 );
		HDMI_WriteI2C_Byte( 0x99, 0x04 );
		HDMI_WriteI2C_Byte( 0x98, DCS_DI );
		HDMI_WriteI2C_Byte( 0x98, *Ptr );
		HDMI_WriteI2C_Byte( 0x98, *( Ptr + 1 ) );
	}
	else
	{
		HDMI_WriteI2C_Byte( 0x9a, 0xe2 );
		HDMI_WriteI2C_Byte( 0x99, ucLen + 6 );
		HDMI_WriteI2C_Byte( 0x98, DCS_DI );
		HDMI_WriteI2C_Byte( 0x98, ucLen );
		HDMI_WriteI2C_Byte( 0x98, 0x00 );
	
		for(i = 0; i < ucLen; i++)
		{
			HDMI_WriteI2C_Byte( 0x98, *Ptr );
			Ptr++;														   
		}
	}
	usleep(1*1000);

	HDMI_WriteI2C_Byte( 0xab, 0x00 );

}


#if 0
#define LPT_DI	0x39
#define SPT_DI	0x15


//RDATA u8 dcsrst[] = {0x05,0x01,0x00};
//*************************************
// Sleep-Out
//*************************************
//RDATA u8 Sleep_Out[] = {0x05,0x11,0x00};
//RDATA u8 Sleep_IN[] = {0x05,0x10,0x00};

//DELAY	120
// Display-On
//*************************************
//RDATA u8 Display_On[] = {0x05,0x29,0x00};


#define CMDNUM  (10)

const u8 shiantong_ic9707_720x1280_dcs0[] = {0x05, 0x01,0x00};

const u8 shiantong_ic9707_720x1280_dcs1[] = {LPT_DI, 0xF0,0x5A,0x59};	

const u8 shiantong_ic9707_720x1280_dcs2[] = {LPT_DI, 0xF1,0xA5,0xA6};

const u8 shiantong_ic9707_720x1280_dcs3[] = {LPT_DI, 0xB4, 0x0E, 0x24, 0x01, 0x1C, 0x08, 0x1D, 0x0C, 0x12, 0x10, 0x04, 0x06, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

const u8 shiantong_ic9707_720x1280_dcs4[] = {LPT_DI, 0xB3, 0x0F, 0x24, 0x01, 0x1C, 0x08, 0x1D, 0x0D, 0x13, 0x11, 0x05, 0x07, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

const u8 shiantong_ic9707_720x1280_dcs5[] = {LPT_DI, 0xB0, 0x76, 0x54, 0x76, 0x57, 0x33, 0x33, 0x14, 0x31, 0x24, 0x00, 0x00, 0x9C, 0x00, 0x00, 0x09};

const u8 shiantong_ic9707_720x1280_dcs6[] = {LPT_DI, 0xB1, 0x53, 0xA0, 0x00, 0x85, 0x0A, 0x00, 0x00, 0x80, 0x00, 0x00, 0x5F};

const u8 shiantong_ic9707_720x1280_dcs7[] = {LPT_DI, 0xB2, 0x37, 0x09, 0x08, 0x8B, 0x08, 0x00, 0x22, 0x00, 0x44, 0xD9};

const u8 shiantong_ic9707_720x1280_dcs8[] = {LPT_DI, 0xB6, 0x83, 0x83};

const u8 shiantong_ic9707_720x1280_dcs9[] = {LPT_DI, 0xB7, 0x01, 0x01, 0x09, 0x0D, 0x11, 0x19, 0x1D, 0x15, 0x00, 0x25, 0x21, 0x00, 0x00, 0x00, 0x00, 0x02, 0xF7, 0x38};

const u8 shiantong_ic9707_720x1280_dcs10[] = {LPT_DI, 0xB8, 0xB8, 0x52, 0x02, 0xCC};

const u8 shiantong_ic9707_720x1280_dcs11[] = {LPT_DI, 0xBA, 0x27, 0xD3};

const u8 shiantong_ic9707_720x1280_dcs12[] = {LPT_DI, 0xBD, 0x43, 0x0E, 0x0E, 0x70, 0x70, 0x32, 0x10};

const u8 shiantong_ic9707_720x1280_dcs13[] = {LPT_DI, 0xC1, 0x00, 0x0F, 0x0E, 0x01, 0x00, 0x36, 0x3A, 0x08};

const u8 shiantong_ic9707_720x1280_dcs14[] = {LPT_DI, 0xC3, 0x02, 0x31};

const u8 shiantong_ic9707_720x1280_dcs15[] = {LPT_DI, 0xC6, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00};

const u8 shiantong_ic9707_720x1280_dcs16[] = {LPT_DI, 0xC7, 0x45, 0x2B, 0x41, 0x00, 0x02};

const u8 shiantong_ic9707_720x1280_dcs17[] = {LPT_DI, 0xC8, 0x7C, 0x63, 0x53, 0x46, 0x42, 0x33, 0x38, 0x22, 0x3C, 0x3C, 0x3D, 0x5C, 0x4B, 0x53, 0x45, 0x41, 0x33, 0x20, 0x06, 0x7C, 0x64, 0x53, 0x46, 0x43, 0x33, 0x38, 0x22, 0x3C, 0x3C, 0x3D, 0x5C, 0x4B, 0x53, 0x45, 0x41, 0x33, 0x20, 0x06};

const u8 shiantong_ic9707_720x1280_dcs18[] = {LPT_DI, 0xD4, 0x00, 0x00, 0x00, 0x32, 0x04, 0x51};

const u8 shiantong_ic9707_720x1280_dcs19[] = {LPT_DI, 0xF1, 0x5A, 0x59};

const u8 shiantong_ic9707_720x1280_dcs20[] = {LPT_DI, 0xF0, 0xA5, 0xA6};

const u8 shiantong_ic9707_720x1280_dcs21[] = {0x05,0x11,0x00};

const u8 shiantong_ic9707_720x1280_dcs22[] = {0x05,0x29,0x00};

int gpio_test_out(unsigned int gpio_chip_num, unsigned int gpio_offset_num,unsigned int gpio_out_val)
{
	FILE *fp;
	char file_name[50];
	unsigned char buf[10];
	unsigned int gpio_num;
	gpio_num = gpio_chip_num * 8 + gpio_offset_num;
	sprintf(file_name, "/sys/class/gpio/export");
	fp = fopen(file_name, "w");
	if (fp == NULL) {
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
	fprintf(fp, "%d", gpio_num);
	fclose(fp);
	sprintf(file_name, "/sys/class/gpio/gpio%d/direction", gpio_num);
	fp = fopen(file_name, "rb+");
	if (fp == NULL) {
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
 	fprintf(fp, "out");
	fclose(fp);
	sprintf(file_name, "/sys/class/gpio/gpio%d/value", gpio_num);
	fp = fopen(file_name, "rb+");
	if (fp == NULL) {
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
	if (gpio_out_val)
	strcpy(buf,"1");
	else
	strcpy(buf,"0");
	fwrite(buf, sizeof(char), sizeof(buf) - 1, fp);
	printf("%s: gpio%d_%d = %s\n", __func__,
	gpio_chip_num, gpio_offset_num, buf);
	fclose(fp);
	sprintf(file_name, "/sys/class/gpio/unexport");
	fp = fopen(file_name, "w");
	if (fp == NULL) {
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
	fprintf(fp, "%d", gpio_num);
	fclose(fp);
	return 0;
}

void Drv_Lcd_Reset(void)
{
//    P10 = 0;
//    msleep(200);
//    P10 = 1;
//    msleep(200);
	gpio_test_out(6,6,0);
	usleep(2000000);
	gpio_test_out(6,6,1);
	usleep(2000000);
}

void Drv_MipiTx_PanelInit(void)
{

	HDMI_WriteI2C_Byte( 0xff, 0xd4 );
    HDMI_WriteI2C_Byte( 0xab, 0x00 );
	HDMI_WriteI2C_Byte( 0xb6, 0x10 );
	HDMI_WriteI2C_Byte( 0xa3, 0x04 );
	HDMI_WriteI2C_Byte( 0x9a, 0xc2 ); 
	HDMI_WriteI2C_Byte( 0x9b, 0x46 );
	HDMI_WriteI2C_Byte( 0x9c, 0x02 );
	HDMI_WriteI2C_Byte( 0x9d, 0x50 );
	HDMI_WriteI2C_Byte( 0x9e, 0x10 );
	HDMI_WriteI2C_Byte( 0x9f, 0x50 );

	usleep(5000);
	Drv_Lcd_Reset();
	
	printf("Start initial panel");

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs0[0], (sizeof(shiantong_ic9707_720x1280_dcs0)/sizeof(shiantong_ic9707_720x1280_dcs0[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs0[1]));
	usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs1[0], (sizeof(shiantong_ic9707_720x1280_dcs1)/sizeof(shiantong_ic9707_720x1280_dcs1[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs1[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs2[0], (sizeof(shiantong_ic9707_720x1280_dcs2)/sizeof(shiantong_ic9707_720x1280_dcs2[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs2[1]));
    usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs3[0], (sizeof(shiantong_ic9707_720x1280_dcs3)/sizeof(shiantong_ic9707_720x1280_dcs3[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs3[1]));
    usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs4[0], (sizeof(shiantong_ic9707_720x1280_dcs4)/sizeof(shiantong_ic9707_720x1280_dcs4[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs4[1]));
    usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs5[0], (sizeof(shiantong_ic9707_720x1280_dcs5)/sizeof(shiantong_ic9707_720x1280_dcs5[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs5[1]));
    usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs6[0], (sizeof(shiantong_ic9707_720x1280_dcs6)/sizeof(shiantong_ic9707_720x1280_dcs6[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs6[1]));
    usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs7[0], (sizeof(shiantong_ic9707_720x1280_dcs7)/sizeof(shiantong_ic9707_720x1280_dcs7[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs7[1]));
    usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs8[0], (sizeof(shiantong_ic9707_720x1280_dcs8)/sizeof(shiantong_ic9707_720x1280_dcs8[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs8[1]));
    usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs9[0], (sizeof(shiantong_ic9707_720x1280_dcs9)/sizeof(shiantong_ic9707_720x1280_dcs9[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs9[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs10[0], (sizeof(shiantong_ic9707_720x1280_dcs10)/sizeof(shiantong_ic9707_720x1280_dcs10[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs10[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs11[0], (sizeof(shiantong_ic9707_720x1280_dcs11)/sizeof(shiantong_ic9707_720x1280_dcs11[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs11[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs12[0], (sizeof(shiantong_ic9707_720x1280_dcs12)/sizeof(shiantong_ic9707_720x1280_dcs12[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs12[1]));
    usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs13[0], (sizeof(shiantong_ic9707_720x1280_dcs13)/sizeof(shiantong_ic9707_720x1280_dcs13[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs13[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs14[0], (sizeof(shiantong_ic9707_720x1280_dcs14)/sizeof(shiantong_ic9707_720x1280_dcs14[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs14[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs15[0], (sizeof(shiantong_ic9707_720x1280_dcs15)/sizeof(shiantong_ic9707_720x1280_dcs15[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs15[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs16[0], (sizeof(shiantong_ic9707_720x1280_dcs16)/sizeof(shiantong_ic9707_720x1280_dcs16[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs16[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs17[0], (sizeof(shiantong_ic9707_720x1280_dcs17)/sizeof(shiantong_ic9707_720x1280_dcs17[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs17[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs18[0], (sizeof(shiantong_ic9707_720x1280_dcs18)/sizeof(shiantong_ic9707_720x1280_dcs18[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs18[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs19[0], (sizeof(shiantong_ic9707_720x1280_dcs19)/sizeof(shiantong_ic9707_720x1280_dcs19[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs19[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs20[0], (sizeof(shiantong_ic9707_720x1280_dcs20)/sizeof(shiantong_ic9707_720x1280_dcs20[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs20[1]));
    usleep(1000);
    
	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs21[0], (sizeof(shiantong_ic9707_720x1280_dcs21)/sizeof(shiantong_ic9707_720x1280_dcs21[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs21[1]));
	usleep(1000);

	Drv_MipiTx_DcsPktWrite(shiantong_ic9707_720x1280_dcs22[0], (sizeof(shiantong_ic9707_720x1280_dcs22)/sizeof(shiantong_ic9707_720x1280_dcs22[0]) - 1), (u8*)&(shiantong_ic9707_720x1280_dcs22[1]));
	usleep(1000);

    printf("Finish initial panel");

} 
#endif

#if 1
#define LPT_DI	0x29


void Drv_MipiTx_PanelInit(void)
{

	HDMI_WriteI2C_Byte( 0xff, 0xd4 );
	HDMI_WriteI2C_Byte( 0xA2, 0x01 ); //LPRX CMD PORT SEL
	
    HDMI_WriteI2C_Byte( 0xab, 0x00 );
	HDMI_WriteI2C_Byte( 0xb6, 0x10 );
	HDMI_WriteI2C_Byte( 0xa3, 0x04 );
	HDMI_WriteI2C_Byte( 0x9a, 0xc2 ); 
	HDMI_WriteI2C_Byte( 0x9b, 0x46 );
	HDMI_WriteI2C_Byte( 0x9c, 0x02 );
	HDMI_WriteI2C_Byte( 0x9d, 0x50 );
	HDMI_WriteI2C_Byte( 0x9e, 0x10 );
	HDMI_WriteI2C_Byte( 0x9f, 0x50 );

	usleep(5*1000);

    printf("\nFinish initial panel");
}

#endif

#endif
