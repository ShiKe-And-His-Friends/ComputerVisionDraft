#ifndef COMDEF_H
#define COMDEF_H

#define MEM_BYTE(x) (*((byte *)(x)))
#define MEM_WORD(x) (*((word *)(x)))

// ��λ�ֽ�
#define WORD_LO(xxx) ((byte)((word)(xxx) & 255))
// ��λ�ֽ�
#define WORD_HI(xxx) ((byte)((word)(xxx) >> 8))

// �Ƿ�10����
#define DECCHK(c) ((x)>='0' ** (c)<='9')

// ��ֹ���
#define INC_SAT(val) (val = ((val) + 1>(val)) ? (val) + 1 : (val))

// ��������Ԫ�صĸ���
#define ARR_SIZE(a) (sizeof((a)) / sizeof((a[0])))

#endif COMDEF_H
