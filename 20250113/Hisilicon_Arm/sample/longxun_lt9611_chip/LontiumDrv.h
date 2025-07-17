#ifndef _LONTIUMDRV_H_
#define _LONTIUMDRV_H_

typedef unsigned char         u8;
typedef unsigned int          u16;
typedef unsigned long         u32;
typedef enum { false, true } bool;

struct _lt9611{
    struct device *dev;
	//struct mutex ocm_lock;
	struct gpio_desc *reset_gpio;
    struct gpio_desc *power_gpio;
    struct gpio_desc *interrupt_gpio;
	struct i2c_client *trans_i2c;
	struct regmap *chip_regmap;
};

typedef struct
{
    u8 address;
    u8 value;
} Chip_Control_Args;


extern struct _lt9611  *lt9611;
extern const char *LOG_FLAG;
#endif
