void LCA_Reconstruct3D::CalUndistortCoor_SSE(const __m256& v_coorx, const __m256& v_coory, __m256& v_dstx, __m256& v_dsty)
{
	__m256 _v_w = _mm256_set1_ps(m_SizeSensor.width);
	__m256i _vi_w = _mm256_set1_epi32(m_SizeSensor.width);
	__m256 _v_h = _mm256_set1_ps(m_SizeSensor.height);

	__m256i _v_zero = _mm256_set1_epi32(0);
	__m256 _vf_zero = _mm256_set1_ps(0.0f);

	__m256 _v_x = v_coorx;
	__m256 _v_y = v_coory;

	__m256 _v_dst_x = _v_x;
	__m256 _v_dst_y = _v_y;

	__m256i _vi_x_dst = _mm256_cvtps_epi32(_v_dst_x);
	__m256i _vi_y_dst = _mm256_cvtps_epi32(_v_dst_y);

	const int n = 2;

	uint32_t table[8];
	uint32_t tableL[8] = { 0 };
	uint32_t tableH[8] = { 0 };

	for (int i = 0; i < n; i++)
	{
		//判断X是否越界
		__m256 b1 = _mm256_cmp_ps(_v_dst_x, _vf_zero, _CMP_LT_OS);
		__m256 b2 = _mm256_cmp_ps(_v_dst_x, _v_w, _CMP_GE_OS);

		__m256 invb1 = _mm256_cmp_ps(_v_dst_x, _vf_zero, _CMP_GE_OS);
		__m256 invb2 = _mm256_cmp_ps(_v_dst_x, _v_w, _CMP_LT_OS);


		__m256 tmp0 = _mm256_and_ps(_mm256_and_ps(invb2, invb1), _v_dst_x);
		__m256 tmp1 = _mm256_and_ps(b2, _mm256_sub_ps(_v_w, _mm256_set1_ps(-1)));
		_v_dst_x = _mm256_add_ps(tmp0, tmp1);


		//判断Y是否越界
		b1 = _mm256_cmp_ps(_v_dst_y, _vf_zero, _CMP_LT_OS);
		b2 = _mm256_cmp_ps(_v_dst_y, _v_h, _CMP_GE_OS);

		invb1 = _mm256_cmp_ps(_v_dst_y, _vf_zero, _CMP_GE_OS);
		invb2 = _mm256_cmp_ps(_v_dst_y, _v_h, _CMP_LT_OS);


		tmp0 = _mm256_and_ps(_mm256_and_ps(invb2, invb1), _v_dst_y);
		tmp1 = _mm256_and_ps(b2, _mm256_sub_ps(_v_h, _mm256_set1_ps(-1)));
		_v_dst_y = _mm256_add_ps(tmp0, tmp1);

		//获取畸变量
		__m256i _vi_x_dst = _mm256_cvtps_epi32(_mm256_floor_ps(_v_dst_x));
		__m256i _vi_y_dst = _mm256_cvtps_epi32(_mm256_floor_ps(_v_dst_y));


		__m256i y_dstLow = _mm256_unpacklo_epi32(_vi_y_dst, _mm256_setzero_si256());
		__m256i y_dstHigh = _mm256_unpackhi_epi32(_vi_y_dst, _mm256_setzero_si256());

		__m256i w_L_Hight = _mm256_unpackhi_epi32(_vi_w, _mm256_setzero_si256());
		__m256i w_L_Low = _mm256_unpacklo_epi32(_vi_w, _mm256_setzero_si256());

		__m256i rowaddressLow = _mm256_mul_epu32(y_dstLow, w_L_Low);
		__m256i rowaddressLoH = _mm256_mul_epu32(y_dstHigh, w_L_Hight);

		__m256i x_dstLow = _mm256_unpacklo_epi32(_vi_x_dst, _mm256_setzero_si256());
		__m256i x_dstHigh = _mm256_unpackhi_epi32(_vi_x_dst, _mm256_setzero_si256());

		__m256i _v_ptrL = _mm256_add_epi64(rowaddressLow, x_dstLow);
		__m256i _v_ptrH = _mm256_add_epi64(rowaddressLoH, x_dstHigh);


		//tableL:0 2 4 6  ---> 0 1 4 5
		//tableL:0 2 4 6  ---> 2 3 6 7

		_mm256_store_si256((__m256i*)tableL, _v_ptrL);
		_mm256_store_si256((__m256i*)tableH, _v_ptrH);
		//_mm256_store_si256((__m256i*)table+4, rowaddressLoH);

		std::cout << "shikeDebug tableL " << tableL[0] << " " << tableL[2] << " " << tableL[4] << " " << tableL[6] << " " << std::endl;
		std::cout << "shikeDebug tableH " << tableH[0] << " " << tableH[2] << " " << tableH[4] << " " << tableH[6] << " " << std::endl;

		__m256 _v_dis_mapx = _mm256_setr_ps(m_pDistMapX[tableL[0]], m_pDistMapX[tableL[2]], m_pDistMapX[tableH[0]], m_pDistMapX[tableH[2]]
			, m_pDistMapX[tableL[4]], m_pDistMapX[tableL[6]], m_pDistMapX[tableH[4]], m_pDistMapX[tableH[6]]);

		__m256 _v_dis_mapy = _mm256_setr_ps(m_pDistMapY[tableL[0]], m_pDistMapY[tableL[2]], m_pDistMapY[tableH[0]], m_pDistMapY[tableH[2]]
			, m_pDistMapY[tableL[4]], m_pDistMapY[tableL[6]], m_pDistMapY[tableH[4]], m_pDistMapY[tableH[6]]);


		//__m128i _vi_y_dst_16 = _mm256_cvtepi32_ph(_vi_y_dst);
		//__m128i _vi_w_dst_16 = _mm256_cvtepi32_epi16(_vi_w);
		//__m128i rowaddress_16 = _mm_mul_epu32(_vi_w_dst_16, _vi_y_dst_16);
		//_mm256_mul_ps();
	/*	__m256i rowaddressL = _mm256_mullo_epi32(_vi_w, _vi_y_dst);
		__m256i rowaddressH = _mm256_mulhi_epu16(_vi_w, _vi_y_dst);*/

		//__m256i _v_ptr = _mm256_add_epi32(rowaddress, _vi_x_dst);
		//__m256i _v_ptr = { 0 };
		//_mm256_store_si256((__m256i*)table, _v_ptr);


		/*__m256 _v_dis_mapx = _mm256_set1_ps(0.0f);
		__m256 _v_dis_mapy = _mm256_set1_ps(0.0f);*/


		//__m256 _v_dis_mapx = _mm256_setr_ps(m_pDistMapX[0], m_pDistMapX[100], m_pDistMapX[100000], m_pDistMapX[0]
		//	, m_pDistMapX[0], m_pDistMapX[0], m_pDistMapX[0], m_pDistMapX[30000]);

		//__m256 _v_dis_mapy = _mm256_setr_ps(m_pDistMapY[0], m_pDistMapY[0], m_pDistMapY[0], m_pDistMapY[0]
		//	, m_pDistMapY[0], m_pDistMapY[0], m_pDistMapY[0], m_pDistMapY[0]);


		/*__m256 _v_dis_mapx = _mm256_setr_ps(m_pDistMapX[table[0]], m_pDistMapX[table[1]], m_pDistMapX[table[2]], m_pDistMapX[table[3]]
		, m_pDistMapX[table[4]], m_pDistMapX[table[5]], m_pDistMapX[table[6]], m_pDistMapX[table[7]]);

		__m256 _v_dis_mapy = _mm256_setr_ps(m_pDistMapY[table[0]], m_pDistMapY[table[1]], m_pDistMapY[table[2]], m_pDistMapY[table[3]]
			, m_pDistMapY[table[4]], m_pDistMapY[table[5]], m_pDistMapY[table[6]], m_pDistMapY[table[7]]);*/

		_v_dst_x = _mm256_sub_ps(_v_x, _v_dis_mapx);
		_v_dst_y = _mm256_sub_ps(_v_y, _v_dis_mapy);
	}

	v_dstx = _v_dst_x;
	v_dsty = _v_dst_y;

	//const int n = 2;				//迭代次数
	//float fXsrc = fCoorSrc[0];		//原始 
	//float fYsrc = fCoorSrc[1];
	//float fXdst = fXsrc;			//校正后初值
	//float fYdst = fYsrc;

	//bool b1 = false, b2 = false;
	//for (int i = 0; i < n; i++)
	//{
	//	//越界判断
	//	if (fXdst < 0)
	//		fXdst = 0;
	//	else if (fXdst >= m_SizeSensor.width)
	//		fXdst = m_SizeSensor.width - 1;
	//	if (fYdst < 0)
	//		fYdst = 0;
	//	else if (fYdst >= m_SizeSensor.height)
	//		fYdst = m_SizeSensor.height - 1;

	//	////越界判断:已验证，速度更慢
	//	//b1 = fXdst < 0;
	//	//b2 = fXdst >= m_SizeSensor.width;
	//	//fXdst = /*(b1 *0 +) */b2 * (m_SizeSensor.width - 1) + ((!b1)&(!b2))*fXdst;
	//	//b1 = fYdst < 0;
	//	//b2 = fYdst >= m_SizeSensor.height;
	//	//fYdst = /*(b1 *0 +) */b2 * (m_SizeSensor.height - 1) + ((!b1)&(!b2))*fYdst;

	//	//获取畸变量
	//	int iPtr = (int)floor(fYdst) * m_SizeSensor.width + (int)floor(fXdst);
	//	fXdst = fXsrc - m_pDistMapX[iPtr];
	//	fYdst = fYsrc - m_pDistMapY[iPtr];
	//}
	//
	////校正
	//fCoorDst[0] = fXdst;
	//fCoorDst[1] = fYdst;
}

void LCA_Reconstruct3D::CalReconstData_1D_SSE_UINT(unsigned int* pProfileRange, LST_PROFILE_XZ* pProfileXZ)
{
	//重建参数
	__m256 ParamM0 = _mm256_set1_ps(m_ParamMatH[0]);
	__m256 ParamM1 = _mm256_set1_ps(m_ParamMatH[1]);
	__m256 ParamM2 = _mm256_set1_ps(m_ParamMatH[2]);
	__m256 ParamM3 = _mm256_set1_ps(m_ParamMatH[3]);
	__m256 ParamM4 = _mm256_set1_ps(m_ParamMatH[4]);
	__m256 ParamM5 = _mm256_set1_ps(m_ParamMatH[5]);
	__m256 ParamM6 = _mm256_set1_ps(m_ParamMatH[6]);
	__m256 ParamM7 = _mm256_set1_ps(m_ParamMatH[7]);
	__m256 ParamM8 = _mm256_set1_ps(m_ParamMatH[8]);

	//用于和Range数据比较大小
	__m256i _v_zero = _mm256_set1_epi32(0);
	//右移位数
	//__m256 vec_precision = _mm256_set1_ps(SUBPIXEL_PRECISION);
	__m256 vec_precision = _mm256_set1_ps(0.00048828125f);
	__m256 vec_roiy = _mm256_set1_ps(m_ROISensor.y);

	int blocksize = 8, block = (m_SizeData.width) / blocksize;

	float x[8] = { 0 };
	float z[8] = { 0 };

	for (int i = 0; i < blocksize * block; i += blocksize)
	{
		//一次读取8个值   256 = 8*32
		//__m128i _v_data = _mm_loadu_si128((const __m128i*)(pProfileRange + i));

		//将16位数据转换为32位数据
		//__m256i _v_data_32i = _mm256_cvtepi16_epi32(_v_data);

		//读取256位的整形（对齐）
		//__m256i _v_data_32i = _mm256_load_si256((const __m256i*)(pProfileRange + i));

		//读取256位的整形（无需对齐）
		__m256i _v_data_32i = _mm256_loadu_si256((const __m256i*)(pProfileRange + i));

		//将32位数据转换为32位浮点数
		__m256 _v_data_32f = _mm256_cvtepi32_ps(_v_data_32i);

		//debug_32_f(_v_data_32f);

		//生成mask:usRangeY > 0 :0xffff,else :0
		//if (usRangeY == 0)	//无效点
		__m256i vec_mask = _mm256_cmpgt_epi32(_v_data_32i, _v_zero);

		//构造fCoorX
		__m256 vec_CoorSrcX = _mm256_setr_ps(i, i + 1.0f, i + 2.0f, i + 3.0f, i + 4.0f, i + 5.0f, i + 6.0f, i + 7.0f);
		//debug_32_f(vec_CoorSrcX);
		 
		/****
			计算Z值的网格化坐标
		****/
		//float fCoorY = static_cast<float>(usRangeY)*SUBPIXEL_PRECISION + m_ROISensor.y;
		//__m256 vec_CoorSrcY = _mm256_add_ps(_mm256_mul_ps(vec_precision, _v_data_32f), vec_roiy);
		//debug_32_f(vec_CoorSrcY);
		__m256 vec_CoorSrcY = _mm256_mul_ps(vec_precision, _v_data_32f);


		//fu,fv为畸变矫正后坐标标量
		__m256 fu = { 0 };
		__m256 fv = { 0 };

		//SSE 版本的畸变矫正函数
		CalUndistortCoor_SSE(vec_CoorSrcX, vec_CoorSrcY, fu, fv);

		__m256 fuu = _mm256_mul_ps(fu, ParamM6);
		__m256 fvv = _mm256_mul_ps(fv, ParamM7);
		//__m256 test = _mm256_add_ps(_mm256_add_ps(fuu, fvv), ParamM8);

		//_mm256_rcp_ps求倒数是近似解，目前能满足精度
		//__m256 fs = _mm256_rcp_ps(_mm256_add_ps(_mm256_add_ps(fuu,fvv), ParamM8));
		__m256 fs = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_add_ps(fuu, fvv), ParamM8));

		fuu = _mm256_mul_ps(fu, ParamM0);
		fvv = _mm256_mul_ps(fv, ParamM1);
		__m256 outx = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(fuu, fvv), ParamM2), fs);

		fuu = _mm256_mul_ps(fu, ParamM3);
		fvv = _mm256_mul_ps(fv, ParamM4);

		__m256 outz = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(fuu, fvv), ParamM5), fs);


		_mm256_store_ps(x, outx);
		_mm256_store_ps(z, outz);

		for (int m = 0; m < 8; m++)
		{
			if (vec_mask.m256i_u32[m] > 0)
			{
				(*(pProfileXZ + i + m)).x = x[m];
				(*(pProfileXZ + i + m)).z = z[m];
			}
			else
			{
				(*(pProfileXZ + i + m)).x = INVALID_FLOAT_L;
				(*(pProfileXZ + i + m)).z = INVALID_FLOAT_L;
			}
		}
	}

}