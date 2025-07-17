#include	"include.h"

#ifndef		_DRVDSCCMD_H
#define		_DRVDSCCMD_H

#if ((LT9211C_MODE_SEL == LVDS_IN_MIPI_OUT)||(LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)||(LT9211C_MODE_SEL == TTL_IN_MIPI_OUT) || (TX_PATTERN_SRC_SEL == MIPITX_PATTERN))

void Drv_MipiTx_PanelInit(void);

#endif

#endif