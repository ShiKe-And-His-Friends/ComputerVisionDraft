
#ifndef		_LT9611_H
#define		_LT9611_H

typedef unsigned char         u8;
typedef unsigned int          u16;
typedef unsigned long         u32;
//typedef enum { false, true } bool;

/******************************** debug *******************/
//#define _pcr_mk_printf_
#define _mipi_Dphy_debug_
#define _htotal_stable_check_
#define _Frequency_Meter_Byte_Clk_

//#define _pattern_test_
//#define cec_on
//#define _enable_read_edid_
/***********************************************************/
//#define   LT9611_ADR       0x76

#define port_a_mipi      0
#define port_b_mipi      1
#define dual_port_mipi   2

//Do not support 3lane
#define lane_cnt_1   1
#define lane_cnt_2   2  
#define lane_cnt_4   0

#define audio_i2s     0
#define audio_spdif   1

#define dsi           0
#define csi           1

#define Non_Burst_Mode_with_Sync_Events 0x00
#define Non_Burst_Mode_with_Sync_Pulses 0x01
#define Burst_Mode                      0x02

#define ac_mode     0
#define dc_mode     1

#define hdcp_disable 0
#define hdcp_enable 1

#define HPD_INTERRUPT_ENABLE            0x01
#define VID_CHG_INTERRUPT_ENABLE    0x02
#define CEC_INTERRUPT_ENABLE             0x03
 
#define RGB888 0x00
#define YUV422 0x01

#define DVI 0x00
#define HDMI 0x01

#define AR_ND 0x00
#define AR_4_3 0x01
#define AR_16_9 0x02 


#define	MPEG_PKT_EN 0x01
#define	AIF_PKT_EN  0x02
#define SPD_PKT_EN	0x04//9611 RXÎªMIPI£¬²»ÐèÒª·¢SPD_PKT
#define AVI_PKT_EN	0x08
#define UD1_PKT_EN	0x10
#define UD0_PKT_EN	0x20

#define CEC_SEND_DONE	0x01
#define CEC_REC_DATA	0x02
#define CEC_NACK	    0x04
#define CEC_ARB_LOST	0x08
#define CEC_ERROR_INITIATOR	0x10
#define CEC_ERROR_FOLLOWER	0x20
#define CEC_WAKE_UP 	0x40


#define	MSG_USRCNTRL_PRESS		0X44
#define	MSG_USRCNTRL_RELEASE	0X45
#define	MSG_OSD_GET				0x46
#define	MSG_OSD_SET				0x47
#define	MSG_PHYADDR_GET			0x83
#define	MSG_PHYADDR_SET			0x84
#define	MSG_STREAM_SET			0X86
#define	MSG_ACTIVESOURCE_GET	0X82
#define	MSG_VENDOR_GET			0x8C
#define	MSG_MENUSTATUS_GET		0X8D
#define	MSG_MENUSTATUS_SET		0X8E
#define	MSG_MEBUSTATUS_ACTIVE	0X00
#define	MSG_POWERSTATUS_GET		0X8F
#define	MSG_POWERSTATUS_SET		0X90
#define	MSG_POWERSTATUS_ON		0X00
#define	MSG_CECVERSION_GET		0x9F
#define	MSG_CECVERSION_SET		0X9E
#define	MSG_CECVERSION_1_4		0X05
#define	MSG_FEATURE_ABRT		0x00
#define	MSG_ABORT		        0xFF
#define	MSG_ABRT_REASON			0x00


#define CEC_ABORT_REASON_0 	0x00 //Unrecoanized op
#define CEC_ABORT_REASON_1 	0x01 //Not in correct mode to respond
#define CEC_ABORT_REASON_2 	0x02 //Cannot provide source
#define CEC_ABORT_REASON_3 	0x03 //Invalid operand
#define CEC_ABORT_REASON_4 	0x04 //Refused
#define CEC_ABORT_REASON_5 	0x05 //Unable to determine


typedef struct Lontium_IC_Mode{
u8 mipi_port_cnt; //1 or 2
u8 mipi_lane_cnt; //1 or 2 or 4
bool mipi_mode;   //dsi or csi
u8 video_mode;    //Non-Burst Mode with Sync Pulses; Non-Burst Mode with Sync Events
bool audio_out;   //i2s or spdif
bool hdmi_coupling_mode;//ac_mode or dc_mode
bool hdcp_encryption; //hdcp_enable or hdcp_diable
bool hdmi_mode;
u8 input_color_space;  //RGB888 or YUV422
}stu_Lontium_IC_Mode;


typedef struct video_timing{
u16 hfp;
u16 hs;
u16 hbp;
u16 hact;
u16 htotal;
u16 vfp;
u16 vs;
u16 vbp;
u16 vact;
u16 vtotal;
bool h_polarity;
bool v_polarity;
u16	vic;
u8 aspact_ratio;  // 0=no data, 1=4:3, 2=16:9, 3=no data.
u32 pclk_khz;
}stu_video_timing;


typedef struct cec_msg{
u8 cec_status;
u8 rx_data_buff[16];
u8 tx_data_buff[18];
u8 logical_address;
u16 physical_address;
u8 destintion;
u8 retries_times; 
bool la_allocation_done;
bool report_physical_address_done;
}stu_cec_msg;


typedef enum PCRFormat
{
	PCR_None,
	PCR_Standard,
  PCR_640x480_60Hz,
	PCR_1024x600_60Hz
}Enum_PCRFormat;


typedef enum
{
 I2S_2CH,
 I2S_8CH,
 SPDIF
}_Audio_Input_Mode;

#define     Audio_Input_Mode    I2S_2CH


typedef enum
{
 Input_RGB888,
 Input_RGB565,
 Input_YCbCr444,
 Input_YCbCr422_16BIT,
 Input_YCbCr422_20BIT,
 Input_YCbCr422_24BIT,
 Input_BT1120_16BIT,
 Input_BT1120_20BIT,
 Input_BT1120_24BIT,
 Input_BT656_8BIT,
 Input_BT656_10BIT,
 Input_BT656_12BIT
}_Video_Input_Mode;

#define Video_Input_Mode Input_RGB888


typedef enum
{
    Output_RGB888,
    Output_YCbCr444,
    Output_YCbCr422_16BIT,
    Output_YCbCr422_20BIT,
    Output_YCbCr422_24BIT
}_Video_Output_Mode;


#define Video_Output_Mode  Output_RGB888


extern void LT9611_Init(void);
extern void LT9611_Chip_ID(void);
extern void LT9611_pattern(void);
extern void LT9611_IRQ_Task(void);
#endif