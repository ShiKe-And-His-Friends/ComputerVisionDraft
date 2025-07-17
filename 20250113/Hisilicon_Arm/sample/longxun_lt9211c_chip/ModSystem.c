#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
  * @project: LT9211C
  * @file: ModSystem.c
  * @author: sxue
  * @company: LONTIUM COPYRIGHT and CONFIDENTIAL
  * @date: 2023.01.29
/******************************************************************************/

#include    "include.h"

void Mod_System_Init(void)
{
    //�������ź�ʱ��
    //P10_PushPull_Mode; //LT2911R_RST_N
    //P11_PushPull_Mode; //LT2911R_RST_N  
    //P13_OpenDrain_Mode;//SCL_CTL
    //P14_OpenDrain_Mode;//SDA_CTL
    //P12_Input_Mode;    //IRQ
    //Ocm_UART0_Timer1Init(115200);//UART_INTI
}


void Mod_LT9211C_Reset(void)
{
    //��������
    //P11 = 0;
    ////Ocm_Timer0_Delay1ms(100);
    //P11 = 1;
    ////Ocm_Timer0_Delay1ms(100);
}
void Mod_ChipID_Read(void)
{
    HDMI_WriteI2C_Byte(0xff,0x81);//register bank
    printf("\nLT9211C Chip ID: 0x%02x 0x%02x 0x%02x", HDMI_ReadI2C_Byte(0x00),HDMI_ReadI2C_Byte(0x01),HDMI_ReadI2C_Byte(0x02));
    
    #if LT9211C_VERSION == U1
        printf("\nLT9211C Code Version: U1");
    #elif   LT9211C_VERSION == U2
        printf("\nLT9211C Code Version: U2");
    #endif
}


#if (LT9211C_MODE_SEL != PATTERN_OUT)

StructChipRx g_stChipRx;
StructChipTx g_stChipTx;
StructVidChkTiming g_stVidChk;
StructChipRxVidTiming g_stRxVidTiming;
u8 g_b1CscTodo = false;


void Mod_SystemTx_SetState(u8 ucState)
{
    if(ucState != g_stChipTx.ucTxState)
    {        
        g_stChipTx.ucTxState = ucState;
        g_stChipTx.b1TxStateChanged = true;
        printf("\nTxState = 0x%02x", ucState);
    }
}

void Mod_SystemRx_SetState(u8 ucState)
{
    u8 ucLastState;
    if(ucState != g_stChipRx.ucRxState)
    {   
        ucLastState = g_stChipRx.ucRxState;
        g_stChipRx.ucRxState = ucState;
        g_stChipRx.b1RxStateChanged = true;
        printf("\nRxState = 0x%02x", ucState);

        // other state-->STATE_HDMIRX_PLAY_BACK,need notify video on
        if(g_stChipRx.ucRxState == STATE_CHIPRX_PLAY_BACK)
        {
            g_stChipRx.pHdmiRxNotify(MIPIRX_VIDEO_ON_EVENT);
        }

        //STATE_HDMIRX_PLAY_BACK-->other state,need notify video off
        if(ucLastState == STATE_CHIPRX_PLAY_BACK)
        {
            g_stChipRx.pHdmiRxNotify(MIPIRX_VIDEO_OFF_EVENT);
        }
    }
}

void Mod_SystemRx_NotifyRegister(void (*pFunction)())
{
    g_stChipRx.pHdmiRxNotify  = pFunction;
}

void Mod_SystemTx_PowerOnInit(void)
{
    memset(&g_stChipTx, 0, sizeof(StructChipTx));
    g_stChipTx.ucTxState = STATE_CHIPTX_POWER_ON;
}

void Mod_SystemRx_PowerOnInit(void)
{
    memset(&g_stChipRx,0 ,sizeof(StructChipRx));
    memset(&g_stVidChk,0 ,sizeof(StructVidChkTiming));
    memset(&g_stRxVidTiming,0 ,sizeof(StructChipRxVidTiming));
    
    g_stChipRx.ucRxState = STATE_CHIPRX_POWER_ON;    

    Mod_SystemRx_NotifyRegister(Mod_System_RxNotifyHandle);
}

void Mod_System_RxNotifyHandle(EnumChipRxEvent ucEvent)
{
    switch (ucEvent)
    {
        case MIPIRX_VIDEO_ON_EVENT:
            g_stChipTx.b1UpstreamVideoReady = true;
            break;
        case MIPIRX_VIDEO_OFF_EVENT:
            g_stChipTx.b1UpstreamVideoReady = false;
            break;
        case MIPIRX_CSC_EVENT:
            g_b1CscTodo = true;
        default:
            break;
    }
}


#endif