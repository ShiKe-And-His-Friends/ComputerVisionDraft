void SearchLight() {
	  
	///////////////////////////////   读图   ///////////////////////////////////////////
	int iNumImgs = 50;
	std::vector<cv::Mat> resimgs;

	std::string imgfilePath = "F://20230106//平面靶标的新5处位置横放//中间//";

	int base = 1021; //1157 -> 137  1021
	for (int k = 0; k < iNumImgs; k++)
	{
		//std::string file = imgfilePath + "StripeImg-" + std::to_string(k) + ".Bmp"; 
		//std::string file = imgfilePath + "0" + std::to_string(k + base) + ".bmp";
		std::string file;
		if (k + base >= 100000) {
			file = imgfilePath + std::to_string(k + base) + ".bmp";
		}
		else if (k + base >= 10000) {
			file = imgfilePath + "0" + std::to_string(k + base) + ".bmp";
		}
		else {
			file = imgfilePath + "00" + std::to_string(k + base) + ".bmp";
		}

		cv::Mat Image = cv::imread(file, CV_LOAD_IMAGE_COLOR);
		if (Image.cols == 0 || Image.rows == 0) {
			std::cout << "The image size is invaliate : " << file << std::endl;
			exit(-3);
		}
		resimgs.emplace_back(Image);
	}
	std::cout << "input photo nums: " << resimgs.size() << std::endl;

	///////////////////////////////   5X5   ///////////////////////////////////////////

	for (int k = 0; k < iNumImgs; k++) {
		cv::Mat raw_image = resimgs[k];
		cv::GaussianBlur(raw_image, raw_image, cv::Size(5, 5), 1.5, 1.5);
	}


	///////////////////////////////   15X1   ///////////////////////////////////////////
	int m_NormalWindSize  = 15;
	int* m_pTempLateWindow_Normal = new int[m_NormalWindSize];

	m_pTempLateWindow_Normal[0] = 2;
	m_pTempLateWindow_Normal[1] = 7;
	m_pTempLateWindow_Normal[2] = 20;
	m_pTempLateWindow_Normal[3] = 44;
	m_pTempLateWindow_Normal[4] = 80;
	m_pTempLateWindow_Normal[5] = 121;
	m_pTempLateWindow_Normal[6] = 155;
	m_pTempLateWindow_Normal[7] = 168;
	m_pTempLateWindow_Normal[8] = 155;
	m_pTempLateWindow_Normal[9] = 121;
	m_pTempLateWindow_Normal[10] = 80;
	m_pTempLateWindow_Normal[11] = 44;
	m_pTempLateWindow_Normal[12] = 20;
	m_pTempLateWindow_Normal[13] = 7;
	m_pTempLateWindow_Normal[14] = 2;
	
	int width = 1088;
	int height = 4320;

	std::vector<int> std_value_top1;
	std::vector<int> std_value_top2;

	std::ofstream dataStream_std_1("..//LCubor3D_LST_CalibrateFiles//std_1.txt", std::ios::binary | std::ios::trunc | std::ios::out);
	std::ofstream dataStream_std_2("..//LCubor3D_LST_CalibrateFiles//std_2.txt", std::ios::binary | std::ios::trunc | std::ios::out);

	// 一帧图片的宽度 4320
	for (int i = 0; i < 2; i++) { //width

		float sum_postion_val = 0;
		float gray_sum = 0;
		int window_start = 0;
		int second_start = 0;
		int max_allsum_gray = 0;
		float max_position_val = 0;
		float second_max_position_val = 0;
		float sub_pixel_index_z = 0;
		float top1_sub_pixel_index_z = 0;
		float top2_sub_pixel_index_z = 0;

		for (int k = 0; k < 1; k++) { //iNumImgs
			cv::Mat raw_image = resimgs[k];

			// cv::imshow("a" , raw_image);
			// cv::waitKey(1000);

			unsigned char* p = nullptr;
			unsigned char* buff_org_data = raw_image.data;

			p = buff_org_data + i;

			max_position_val = 0;
			top1_sub_pixel_index_z = 0;
			top2_sub_pixel_index_z = 0;
			second_max_position_val = 0;
			window_start = 0;
			second_start = 0;

			//列方向 1088
			int start_index_part = 480;
			for (int j = start_index_part; j < 600; j++) {

				//列卷积 15*1
				sum_postion_val = 0;
				gray_sum = 0;
				sub_pixel_index_z = 0;
				
				for (int t = 0; t < m_NormalWindSize; t++)
				{
					int val = raw_image.at<uchar>((j - m_NormalWindSize / 2 + t), i);
					// int val = p[(j - m_NormalWindSize / 2 + t) * width];
					sum_postion_val += m_pTempLateWindow_Normal[t] * val;

					sub_pixel_index_z +=  (val *  (j - m_NormalWindSize / 2 + t));
					gray_sum += val;
				}

				if (sum_postion_val > max_position_val)
				{
					//std::cout << "" << (sum_postion_val) << " Y " << j << " ";
					max_position_val = sum_postion_val;
					window_start = j;
					max_allsum_gray = sum_postion_val / gray_sum;

					top1_sub_pixel_index_z = sub_pixel_index_z / (gray_sum + 0.0001);
				}

				if (sum_postion_val < max_position_val && sum_postion_val > second_max_position_val) {
					second_max_position_val = sum_postion_val;
					second_start = j;
					//second_max_allsum_gray = sum_postion_val / gray_sum;
					top2_sub_pixel_index_z = sub_pixel_index_z / (gray_sum + 0.0001);
				}

			}
		
			//最大值 次大值
			std_value_top1.push_back(window_start);
			std_value_top2.push_back(second_start);

			// 计算Z值
			float* H1 = new float[9];
			H1[3] = -1.54162281e-05;
			H1[4] = 1.41186779e-02;
			H1[5] = -6.21927500e+00;
			H1[6] = -3.05965750e-06;
			H1[7] = -9.85387305e-05;
			H1[8] = 1.0;
			int step = 600;

			float z1 = (H1[3] * i + H1[4] * (top1_sub_pixel_index_z + step) + H1[5]) / (H1[6] * i + H1[7] * (top1_sub_pixel_index_z + step) + H1[8]);
			float z2 = (H1[3] * i + H1[4] * (top2_sub_pixel_index_z + step) + H1[5]) / (H1[6] * i + H1[7] * (top2_sub_pixel_index_z + step) + H1[8]);
			// 打印坐标数值
			std::cout << "列起始坐标最大" << (window_start) << " 次大值" << second_start << " 权重坐标最大" << top1_sub_pixel_index_z << " 坐标次大" << top2_sub_pixel_index_z << " Z1值" << z1 << " Z2值" << z2 << std::endl;

		}
	

		//计算标准差 top1 top2
		//top1
		double sum_std_value = 0;
		for (int i = 0; i < std_value_top1.size(); i++) {
			int value = std_value_top1[i];
			sum_std_value += value;
		}
		double equal_std_value = sum_std_value / std_value_top1.size();
		sum_std_value = 0;
		for (int i = 0; i < std_value_top1.size(); i++) {
			sum_std_value += (std_value_top1[i] - equal_std_value) * (std_value_top1[i] - equal_std_value);
		}
		double result_std_1 = std::pow(sum_std_value, 0.5);
		// top2
		sum_std_value = 0;
		for (int i = 0; i < std_value_top2.size(); i++) {
			int value = std_value_top2[i];
			sum_std_value += value;
		}
		equal_std_value = sum_std_value / std_value_top2.size();
		sum_std_value = 0;
		for (int i = 0; i < std_value_top2.size(); i++) {
			sum_std_value += (std_value_top2[i] - equal_std_value) * (std_value_top2[i] - equal_std_value);
		}
		double result_std_2 = std::pow(sum_std_value, 0.5);
		// 打印标准差
		std::cout << "标准差 " << result_std_1 << " " << result_std_2 << std::endl;
		
		dataStream_std_1 << result_std_1 << "\n";
		dataStream_std_2 << result_std_2 << "\n";

		std_value_top1.clear();
		std_value_top2.clear();

		// cv::imshow("a" , raw_image);
		// cv::waitKey(10);
	}


	///////////////////////////////   结果   ///////////////////////////////////////////
	dataStream_std_1.close();
	dataStream_std_2.close();

}
//做5X5滤波处理
				
//TODO 整形
cv::GaussianBlur(m_LaserStripeImgSrcImg, m_ProcessedStripImg, cv::Size(5, 5), 1.5, 1.5);

cv::Mat Mask_DRowCol = (cv::Mat_<int>(5, 5) << 
	15, 29, 36, 29, 15
	, 29, 56, 70, 56, 29
	, 36, 70, 87, 70, 36
	, 29, 56, 70, 56, 29
	, 15, 29, 36, 29, 15);

/*cv::Mat Mask_DRowCol = (cv::Mat_<float>(5, 5) <<
	14.76487000315988, 28.75803991690155, 35.91444562492774, 28.75803991690155, 14.76487000315988,
	28.75803991690155, 56.01301330015833, 69.95178830927892, 56.01301330015833, 28.75803991690155,
	35.91444562492774, 69.95178830927892, 87.35921171468807, 69.95178830927892, 35.91444562492774,
	28.75803991690155, 56.01301330015833, 69.95178830927892, 56.01301330015833, 28.75803991690155,
	14.76487000315988, 28.75803991690155, 35.91444562492774, 28.75803991690155, 14.76487000315988);*/

/*cv::Mat Mask_DRowCol = (cv::Mat_<int>(2, 2) <<
	1 ,0 ,0 ,1);*/

cv::Mat filter_result_mat(100 ,100 , CV_32FC1);

cv::filter2D(m_LaserStripeImgSrcImg, m_LaserStripeImgSrcImg, CV_16UC1, Mask_DRowCol);//CV_32FC1  CV_16SC1 CV_16UC1 m_LaserStripeImgSrcImg.depth()

m_LaserStripeImgSrcImg.convertTo(m_LaserStripeImgSrcImg, CV_8UC1 ,1.0 / 256.0);

cv::imshow("1" , m_LaserStripeImgSrcImg);
cv::waitKey(1000);

cv::imwrite("..//LCubor3D_LST_TestData//temp//原始5X5高斯核.bmp", m_ProcessedStripImg);
cv::imwrite("..//LCubor3D_LST_TestData//temp//标准5X5高斯核.bmp", m_LaserStripeImgSrcImg);
