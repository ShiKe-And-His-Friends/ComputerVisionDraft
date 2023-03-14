void showRawPhoto() {
	const char* rawFileName = "C:\\Users\\s05559\\Documents\\00Work_Plane_Groupby_Data\\202303\\Image_2023_0309_1704_28_768-009874.raw";
	FILE* fp = NULL;
	int ret = 0, width = 4320, height = 1088;
	unsigned char* pRawData = (unsigned char*)malloc(width * height * sizeof(unsigned char));

	if (NULL == (fp = fopen(rawFileName, "rb"))) {
		std::cout << "open raw file failure." << std::endl;
		return;
	}

	ret = fread(pRawData, sizeof(unsigned char) * width * height, 1, fp);
	if (ret != 1) {
		std::cout << "read raw file failure." << std::endl;
		return;
	}

	cv::Mat img(cv::Size(width ,height) ,CV_8UC1 ,pRawData);
	cv::Mat img2(cv::Size(width ,height) ,CV_8UC3 ,cv::Scalar(0));
	cv::cvtColor(img ,img2 ,CV_BayerGB2BGR);
	
	cv::namedWindow("Raw Photo" ,1);
	cv::imshow("Raw Photo" ,img);
	cv::imwrite("..//LCubor3D_LST_TestData//extract_line_img//raw_photo.png", img);
	cv::waitKey(2000);
	
	/****
		Bayer=>RGB (CV_BayerBG2BGR, CV_BayerGB2BGR, CV_BayerRG2BGR, CV_BayerGR2BGR, CV_BayerBG2RGB, CV_BayerRG2BGR, CV_BayerGB2RGB, CV_BayerGR2BGR, CV_BayerRG2RGB, CV_BayerBG2BGR, CV_BayerGR2RGB, CV_BayerGB2BGR)
	***/
}