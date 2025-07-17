/******************************************************************************
  * @project: LT9211C
  * @file: ModMipiRpt.h
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/
#include	"include.h"

#ifndef		_MODMIPIRPT_H
#define		_MODMIPIRPT_H

#if (LT9211C_MODE_SEL == MIPI_REPEATER)


#define     MIPIRPT_INPUT_PORT          PORTA            //PORTA/PORTB/DOU_PORT
#define     MIPIRPT_OUTPUT_PORT         DOU_PORT            //PORTA/PORTB/DOU_PORT
#define     MIPIRPT_LANE_NUM            MIPIRPT_4LANE       //MIPIRPT_4LANE/MIPIRPT_1LANE/MIPIRPT_2LANE/MIPIRPT_3LANE
#define     MIPIRPT_PORT_SWAP           DISABLED            //ENABLED/DISABLED
#define     MIPIRPT_PORT_COPY           NO_COPY             //PORTA/PORTB/NO_COPY
#define     MIPIRPT_LPCMD_SEND          ENABLED         //ENABLED/DISABLED













extern void Mod_MipiRpt_Handler();



#endif

#endif