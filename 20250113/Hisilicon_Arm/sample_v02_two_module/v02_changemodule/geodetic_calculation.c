#include "geodetic_calculation.h"

// 角度转弧度
double deg_to_rad(double degrees) {
    return degrees * M_PI / 180.0;
}

// 弧度转角度
double rad_to_deg(double radians) {
    return radians * 180.0 / M_PI;
}

// 计算卯酉圈曲率半径
double calculate_N(double latitude_rad) {
    return A / sqrt(1 - E_SQ * pow(sin(latitude_rad), 2));
}

// 经纬度海拔转ECEF坐标
void lla_to_ecef(double lat_rad, double lon_rad, double alt,
    double* x, double* y, double* z) {
    double N = calculate_N(lat_rad);
    *x = (N + alt) * cos(lat_rad) * cos(lon_rad);
    *y = (N + alt) * cos(lat_rad) * sin(lon_rad);
    *z = (N * (1 - E_SQ) + alt) * sin(lat_rad);
}

// ECEF坐标转经纬度海拔（迭代法）
void ecef_to_lla(double x, double y, double z,
    double* lat_rad, double* lon_rad, double* alt) {
    double p = sqrt(x * x + y * y);
    double theta = atan2(z * A, p * sqrt(A * A - (A * A - (A * A * E_SQ))));

    double sin_theta = sin(theta);
    double cos_theta = cos(theta);

    // 迭代计算纬度
    double lat = atan2(z + E_SQ / (1 - E_SQ) * A * pow(sin_theta, 3),
        p - E_SQ * A * pow(cos_theta, 3));

    // 迭代收敛（通常3-5次迭代）
    for (int i = 0; i < 5; i++) {
        double N = calculate_N(lat);
        lat = atan2(z + E_SQ * N * sin(lat), p);
    }

    double N = calculate_N(lat);
    double alt_temp = p / cos(lat) - N;

    *lat_rad = lat;
    *lon_rad = atan2(y, x);
    *alt = alt_temp;
}

// 根据翻滚角、俯仰角和偏航角计算目标点
void calculate_target_point(double x1_deg, double y1_deg, double z1,
    double theta1_deg, double theta2_deg, double theta3_deg,
    double L, double* x2_deg, double* y2_deg, double* z2) {
    // 转换为弧度
    double x1_rad = deg_to_rad(x1_deg);
    double y1_rad = deg_to_rad(y1_deg);
    double theta1_rad = deg_to_rad(theta1_deg); // 俯仰角
    double theta2_rad = deg_to_rad(theta2_deg); // 翻滚角
    double theta3_rad = deg_to_rad(theta3_deg); // 偏航角

    // 计算ECEF坐标
    double X1, Y1, Z1;
    lla_to_ecef(y1_rad, x1_rad, z1, &X1, &Y1, &Z1);

    // 计算方向向量（东北天坐标系）
    // 偏航角转换（正南为0度，调整为ENU坐标系的方位角）
    double azimuth_rad = - theta3_rad; // ENU方位角从正北顺时针

    // 计算东北天坐标系中的位移分量
    double east = L * cos(theta1_rad) * sin(azimuth_rad);
    double north = L * cos(theta1_rad) * cos(azimuth_rad);
    double up = -L * sin(theta1_rad); // 俯仰角朝上为负，朝下为正

    // 考虑翻滚角的影响（简化模型）
    double temp_east = east;
    east = temp_east * cos(theta2_rad) - up * sin(theta2_rad);
    up = temp_east * sin(theta2_rad) + up * cos(theta2_rad);

    // ENU到ECEF的转换矩阵
    double sin_lat = sin(y1_rad);
    double cos_lat = cos(y1_rad);
    double sin_lon = sin(x1_rad);
    double cos_lon = cos(x1_rad);

    // 计算ECEF位移
    double dX = -east * sin_lon - north * sin_lat * cos_lon + up * cos_lat * cos_lon;
    double dY = east * cos_lon - north * sin_lat * sin_lon + up * cos_lat * sin_lon;
    double dZ = north * cos_lat + up * sin_lat;

    // 计算目标点ECEF坐标
    double X2 = X1 + dX;
    double Y2 = Y1 + dY;
    double Z2 = Z1 + dZ;

    // 转换回经纬度
    double lat2_rad, lon2_rad;
    ecef_to_lla(X2, Y2, Z2, &lat2_rad, &lon2_rad, z2);

    // 转换为度
    *x2_deg = rad_to_deg(lon2_rad);
    *y2_deg = rad_to_deg(lat2_rad);
}
