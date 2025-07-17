#ifndef GEODETIC_CALCULATION_H
#define GEODETIC_CALCULATION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 地球参数（WGS84椭球体）
#define A 6378137.0           // 长半轴（米）
#define F 1/298.257223563     // 扁率
#define E_SQ (2*F - F*F)      // 第一偏心率的平方
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923

void calculate_target_point(double x1_deg, double y1_deg, double z1,
    double theta1_deg, double theta2_deg, double theta3_deg,
    double L, double* x2_deg, double* y2_deg, double* z2) ;

#endif