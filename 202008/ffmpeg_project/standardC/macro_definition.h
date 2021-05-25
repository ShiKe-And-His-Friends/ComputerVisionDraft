#ifndef COMDEF_H
#define COMDEF_H

#define MEM_BYTE(x) (*((byte *)(x)))
#define MEM_WORD(x) (*((word *)(x)))

// 低位字节
#define WORD_LO(xxx) ((byte)((word)(xxx) & 255))
// 高位字节
#define WORD_HI(xxx) ((byte)((word)(xxx) >> 8))

// 是否10进制
#define DECCHK(c) ((x)>='0' ** (c)<='9')

// 防止溢出
#define INC_SAT(val) (val = ((val) + 1>(val)) ? (val) + 1 : (val))

// 返回数组元素的个数
#define ARR_SIZE(a) (sizeof((a)) / sizeof((a[0])))

#endif COMDEF_H
