ifeq ($(EXTDRV_PARAM_FILE), )
    EXTDRV_PARAM_FILE:=../Makefile.param
    include $(EXTDRV_PARAM_FILE)
endif

SRCS := my_key_event.c

INC := -I$(REL_INC)
INC += -I$(OSAL_ROOT)/linux/kernel/ot_media
INC += -I$(OSAL_ROOT)/include

INC += -DOT_GPIO_I2C
INC+=-I$(PWD)/../gpio-i2c-ex

EXTDRV_CFLAGS += $(INC)

#*************************************************************************
TARGET := ot_my_key_event

#*************************************************************************
# compile linux or liteos
include $(PWD)/../Make.$(OSTYPE)
