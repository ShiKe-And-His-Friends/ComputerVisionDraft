#ifndef ZOOM_INOUT_THREAD
#define ZOOM_INOUT_THREAD

#include "uart_camera_ctrl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

// 定义相机控制状态
typedef enum {
    ZOOM_STOP = 0,    // 停止变倍
    ZOOM_INCREASE,    // 倍率增加
    ZOOM_DECREASE     // 倍率减少
} ZoomState;

// 定义倍率与坐标的映射表
typedef struct {
    float magnification;  // 倍率
    unsigned int coord;   // 坐标值（0xpqrs）
} FocusPoint;

// 相机控制全局变量
static ZoomState zoom_state = ZOOM_STOP;     // 当前变倍状态
static float current_magnification = 1.0f;   // 当前倍率
static pthread_mutex_t zoom_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t zoom_thread;                // 变倍控制线程

// 倍率-坐标映射表
static const FocusPoint focus_points[] = {
    {1.0f,  0x0000},
    {1.1f,  0x01ec},
    {1.2f,  0x03d9},
    {1.3f,  0x06ec},
    {1.4f,  0x08b2},
    {1.5f,  0x0a1f},
    {1.6f,  0x0c2e},
    {1.7f,  0x0eae},
    {1.8f,  0x1134},
    {1.9f,  0x13b4},
    {2.0f,  0x1605},
    {2.1f,  0x1779},
    {2.2f,  0x18ed},
    {2.3f,  0x1a28},
    {2.4f,  0x1b5d},
    {2.5f,  0x1c7a},
    {2.6f,  0x1d8b},
    {2.7f,  0x1e7f},
    {2.8f,  0x1f62},
    {2.9f,  0x2043},
    {3.0f,  0x2124},
    {3.1f,  0x21fe},
    {3.2f,  0x22bd},
    {3.3f,  0x237f},
    {3.4f,  0x23af},
    {3.5f,  0x23e1},
    {3.6f,  0x2412},
    {3.7f,  0x2444},
    {3.8f,  0x24f0},
    {3.9f,  0x25f1},
    {4.0f,  0x26f3},
    {4.1f,  0x27f3},
    {4.2f,  0x28d9},
    {4.3f,  0x2955},
    {4.4f,  0x29d1},
    {4.5f,  0x2a39},
    {4.6f,  0x2a9e},
    {4.7f,  0x2b04},
    {4.8f,  0x2b6c},
    {4.9f,  0x2bd2},
    {5.0f,  0x2c3a},
    {5.1f,  0x2ca0},
    {5.2f,  0x2d06},
    {5.3f,  0x2d6e},
    {5.4f,  0x2dd4},
    {5.5f,  0x2e3b},
    {5.6f,  0x2e88},
    {5.7f,  0x2ed6},
    {5.8f,  0x2f25},
    {5.9f,  0x2f71},
    {6.0f,  0x2fc0},
    {6.1f,  0x300d},
    {6.2f,  0x305b},
    {6.3f,  0x30a8},
    {6.4f,  0x30f7},
    {6.5f,  0x3145},
    {6.6f,  0x3192},
    {6.7f,  0x31cf},
    {6.8f,  0x320e},
    {6.9f,  0x324d},
    {7.0f,  0x328b},
    {7.1f,  0x32c9},
    {7.2f,  0x3307},
    {7.3f,  0x3345},
    {7.4f,  0x3383},
    {7.5f,  0x33c2},
    {7.6f,  0x33ff},
    {7.7f,  0x343e},
    {7.8f,  0x3471},
    {7.9f,  0x34a5},
    {8.0f,  0x34d9},
    {8.1f,  0x350d},
    {8.2f,  0x3541},
    {8.3f,  0x3575},
    {8.4f,  0x35a8},
    {8.5f,  0x35dd},
    {8.6f,  0x3610},
    {8.7f,  0x3643},
    {8.8f,  0x3678},
    {8.9f,  0x36a1},
    {9.0f,  0x36c9},
    {9.1f,  0x36f3},
    {9.2f,  0x371b},
    {9.3f,  0x3744},
    {9.4f,  0x376d},
    {9.5f,  0x3797},
    {9.6f,  0x37be},
    {9.7f,  0x37e8},
    {9.8f,  0x3811},
    {9.9f,  0x3839},
    {10.0f,  0x3862},
    {10.1f,  0x3876},
    {10.2f,  0x388c},
    {10.3f,  0x38a0},
    {10.4f,  0x38b5},
    {10.5f,  0x38c9},
    {10.6f,  0x38dd},
    {10.7f,  0x38f2},
    {10.8f,  0x3906},
    {10.9f,  0x391c},
    {11.0f,  0x3930},
    {11.1f,  0x3944},
    {11.2f,  0x3959},
    {11.3f,  0x396d},
    {11.4f,  0x3982},
    {11.5f,  0x3996},
    {11.6f,  0x39aa},
    {11.7f,  0x39c0},
    {11.8f,  0x39d3},
    {11.9f,  0x39e9},
    {12.0f,  0x39fd},
    {12.1f,  0x3a0c},
    {12.2f,  0x3a1c},
    {12.3f,  0x3a2d},
    {12.4f,  0x3a3d},
    {12.5f,  0x3a4d},
    {12.6f,  0x3a5c},
    {12.7f,  0x3a6c},
    {12.8f,  0x3a7d},
    {12.9f,  0x3a8d},
    {13.0f,  0x3a9c},
    {13.1f,  0x3aac},
    {13.3f,  0x3acd},
    {13.5f,  0x3aec},
    {13.7f,  0x3b0c},
    {13.9f,  0x3b2c},
    {14.0f,  0x3b3c},
    {14.1f,  0x3b4c},
    {14.3f,  0x3b6c},
    {14.5f,  0x3b8c},
    {14.7f,  0x3bab},
    {14.8f,  0x3bbc},
    {15.0f,  0x3bdc},
    {15.2f,  0x3bfb},
    {15.4f,  0x3c1c},
    {15.6f,  0x3c3b},
    {15.8f,  0x3c5c},
    {16.0f,  0x3c7b},
    {16.4f,  0x3ca0},
    {16.8f,  0x3cc7},
    {17.0f,  0x3cd9},
    {17.4f,  0x3cfe},
    {17.8f,  0x3d25},
    {18.0f,  0x3d37},
    {18.4f,  0x3d5c},
    {18.8f,  0x3d83},
    {19.0f,  0x3d95},
    {19.3f,  0x3db2},
    {19.7f,  0x3dd7},
    {20.0f,  0x3df4},
    {20.5f,  0x3e10},
    {21.0f,  0x3e2d},
    {21.5f,  0x3e49},
    {22.0f,  0x3e66},
    {22.5f,  0x3e82},
    {23.0f,  0x3e9f},
    {23.5f,  0x3ecc},
    {24.0f,  0x3efa},
    {24.5f,  0x3f7d},
    {25.0f,  0x4000},
    {25.5f,  0x412b},
    {26.0f,  0x422f},
    {26.5f,  0x4333},
    {27.0f,  0x441d},
    {27.5f,  0x4507},
    {28.0f,  0x45f1},
    {28.5f,  0x46db},
    {29.0f,  0x47ab},
    {29.5f,  0x487c},
    {30.0f,  0x4925},
};

#define FOCUS_POINT_COUNT (sizeof(focus_points) / sizeof(focus_points[0]))
#define ZOOM_INTERVAL_MS 12  // 变倍间隔时间（毫秒）

void set_zoom_state(ZoomState state);

float get_current_magnification();

int init_colorcamera_controller();

#endif