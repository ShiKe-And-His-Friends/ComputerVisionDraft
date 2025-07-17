include ../Makefile.param

CFLAGS += -Wall $(INC_FLAGS)
CFLAGS += -lstdc++
CFLAGS += -lpthread -lm -ldl
CFLAGS += $(LIBS_CFLAGS)
CFLAGS += $(LIBS_EXEC_CFLAGS)

########### SRCS & INC ############
SMP_SRCS := $(wildcard $(PWD)/*.c)

TARGET := longxun_lt9611_chip_main

TARGET_PATH := $(PWD)

# compile linux or LITEOS
include $(PWD)/../$(ARM_ARCH)_$(OSTYPE).mak
