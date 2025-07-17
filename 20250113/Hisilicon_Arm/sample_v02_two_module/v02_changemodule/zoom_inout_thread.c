#include "zoom_inout_thread.h"

// 根据倍率查找坐标值
static unsigned int find_coord_by_magnification(float mag) {
    // 二分查找（假设表格按倍率升序排列）
    int left = 0, right = FOCUS_POINT_COUNT - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (focus_points[mid].magnification == mag) {
            return focus_points[mid].coord;
        } else if (focus_points[mid].magnification < mag) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    // 未找到精确匹配，返回最接近的较低值
    if (left > 0) return focus_points[left-1].coord;
    return focus_points[0].coord;  // 默认返回最小值
}

// 控制相机变倍到指定倍率
static int set_camera_magnification(float magnification) {
    if (magnification < 1.0f || magnification > 30.0f) {
        return -1;  // 倍率超出范围
    }
    
    // 查找对应的坐标值
    unsigned int coord = find_coord_by_magnification(magnification);
    
    // 构造指令字符串（格式："81 01 04 48 0p 0q 0r 0s FF"）
    char cmd_buffer_zoom_info[50];
    char cmd_buffer_focus_info[50];
    sprintf(cmd_buffer_zoom_info, "81 01 04 47 %02X %02X %02X %02X FF",
            (coord >> 12) & 0xFF,  // p
            (coord >> 8) & 0xFF,   // q
            (coord >> 4) & 0xFF,   // r
            coord & 0xFF);         // s
    sprintf(cmd_buffer_focus_info, "81 01 04 48 %02X %02X %02X %02X FF",
            (coord >> 12) & 0xFF,  // p
            (coord >> 8) & 0xFF,   // q
            (coord >> 4) & 0xFF,   // r
            coord & 0xFF);         // s
    
    // 构造参数字符串数组
    char *argv_zoom[3] = {
        "2",          // 指令类型
        cmd_buffer_zoom_info,   // 指令数据
        NULL          // 数组结束标记
    };
    char *argv_focus[3] = {
        "2",          // 指令类型
        cmd_buffer_focus_info,   // 指令数据
        NULL          // 数组结束标记
    };
    
    // 发送指令（波特率固定为9600）
    // int ret1 =  send_uart_commond(argv_focus, 0);
    // int ret2 =  send_uart_commond(argv_zoom, 0);    
    // return (ret1 == 0 && ret2 == 0) ? 0 : -1;

    int ret =  send_uart_commond(argv_zoom, 0);
    return ret;
}

// 变倍控制线程函数
static void* zoom_control_thread(void* arg) {
    while (1) {
        // 锁定互斥锁，获取当前状态和倍率
        pthread_mutex_lock(&zoom_mutex);
        ZoomState state = zoom_state;
        float magnification = current_magnification;
        pthread_mutex_unlock(&zoom_mutex);
        
        // 根据状态执行相应操作
        switch (state) {
            case ZOOM_INCREASE: {
                // 增加倍率（每次增加0.1x，直到最大值）
                magnification += 0.1f;
                if (magnification > 30.0f) {
                    magnification = 30.0f;
                    zoom_state = ZOOM_STOP;
                    break;
                }
                
                // 设置新倍率
                if (set_camera_magnification(magnification) == 0) {
                    // 更新当前倍率
                    pthread_mutex_lock(&zoom_mutex);
                    current_magnification = magnification;
                    pthread_mutex_unlock(&zoom_mutex);
                    printf("Zoom increased to %.1fx\n", magnification);
                } else {
                    printf("Failed to increase zoom\n");
                }
                break;
            }
            
            case ZOOM_DECREASE: {
                // 减少倍率（每次减少0.1x，直到最小值）
                magnification -= 0.1f;
                if (magnification < 1.0f) {
                    magnification = 1.0f;
                    zoom_state = ZOOM_STOP;
                    break;
                }
                
                // 设置新倍率
                if (set_camera_magnification(magnification) == 0) {
                    // 更新当前倍率
                    pthread_mutex_lock(&zoom_mutex);
                    current_magnification = magnification;
                    pthread_mutex_unlock(&zoom_mutex);
                    printf("Zoom decreased to %.1fx\n", magnification);
                } else {
                    printf("Failed to decrease zoom\n");
                }
                break;
            }
            
            case ZOOM_STOP:
            default:
                // 停止状态，不执行任何操作
                break;
        }
        
        // 休眠指定时间
        usleep(ZOOM_INTERVAL_MS * 1000); // 转换为微秒
        usleep(1);
    }
    
    return NULL;
}

// 设置变倍状态
void set_zoom_state(ZoomState state) {
    pthread_mutex_lock(&zoom_mutex);
    zoom_state = state;
    pthread_mutex_unlock(&zoom_mutex);
}

// 获取当前倍率
float get_current_magnification() {
    float mag;
    pthread_mutex_lock(&zoom_mutex);
    mag = current_magnification;
    pthread_mutex_unlock(&zoom_mutex);
    return mag;
}

// 初始化相机控制线程
int init_colorcamera_controller() {

    zoom_state = ZOOM_STOP;

    // 创建变倍控制线程
    int ret = pthread_create(&zoom_thread, NULL, zoom_control_thread, NULL);
    if (ret != 0) {
        printf("Failed to create zoom control thread\n");
        return -1;
    }
    
    // 设置线程为分离状态，自动释放资源
    pthread_detach(zoom_thread);
    return 0;
}

// 示例：主线程控制相机变倍
// int main() {
//     // 初始化相机控制器
//     if (init_colorcamera_controller() != 0) {
//         return 1;
//     }
    
//     printf("Camera controller initialized. Current zoom: %.1fx\n", get_current_magnification());
//     printf("Press '+' to zoom in, '-' to zoom out, 's' to stop, 'q' to quit\n");
    
//     // 主循环，处理用户输入
//     char cmd;
//     while ((cmd = getchar()) != 'q') {
//         switch (cmd) {
//             case '+':
//                 set_zoom_state(ZOOM_INCREASE);
//                 printf("Colorcamera Zooming in...\n");
//                 break;
//             case '-':
//                 set_zoom_state(ZOOM_DECREASE);
//                 printf("Colorcamera Zooming out...\n");
//                 break;
//             case 's':
//                 set_zoom_state(ZOOM_STOP);
//                 printf("Colorcamera Zoom stopped at %.1fx\n", get_current_magnification());
//                 break;
//         }
//         // 消耗换行符
//         if (cmd != '\n') {
//             while (getchar() != '\n');
//         }
//     }
    
//     // 退出前停止变倍
//     set_zoom_state(ZOOM_STOP);
//     printf("Exiting...\n");
    
//     return 0;
// }