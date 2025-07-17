#include	"include.h"

#ifndef     _MODPATTERN_H
#define     _MODPATTERN_H

#if (LT9211C_MODE_SEL  == PATTERN_OUT)


/******************* Lvds Output Config ********************/

typedef struct video_pattern_timing
{
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
	
	u8 ucFrameRate;
};


#define     NO_PATTERN                    0
#define     PTN_640x480_15                1
#define     PTN_640x480_60                2
#define     PTN_720x480_60                3
#define     PTN_1920x720_60               4
#define     PTN_1920x1080_30              5
#define     PTN_1920x1080_60              6
#define     PTN_1920x1080_90              7
#define     PTN_1920x1080_100             8
#define     PTN_2560x1440_60              9
#define     PTN_3840x2160_30              10
#define     PTN_3840x2160_60              11
#define     PTN_1280x720_60               12    
#define     PTN_720x288_50                13
#define     PTN_1024x600_60               14
#define     PTN_440x1920_60               15
extern void Mod_ChipTx_PtnOut();


#endif

#endif
