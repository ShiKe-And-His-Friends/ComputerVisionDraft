#define BMP_HEADER_LEN 54
#define SAVE_DATA 0

unsigned char* ReadDataFromRawFile(char *m_strFilePath ,long *filesize) {
	std::cout << "file " << m_strFilePath << std::endl;
	FILE* fp;
	fp = fopen(m_strFilePath ,"rb");
	if (fp == NULL) {
		std::cout << "打开RAW文件失败" << std::endl;
		return NULL;
	}
	fpos_t startpos, endpos;
	fseek(fp ,0 ,SEEK_END);
	fgetpos(fp ,&endpos);
	fseek(fp ,0 ,SEEK_SET);
	fgetpos(fp ,&startpos);

	long filelen = (long)(endpos - startpos);

	unsigned char* bTemp = NULL;
	bTemp = (unsigned char*)malloc(filelen);
	if (bTemp == NULL) {
		fclose(fp);
		return NULL;
	}
	memset(bTemp ,0 ,filelen);
	fread(bTemp ,filelen ,1 ,fp);
	fclose(fp);
	*filesize = filelen;

	return bTemp;
}

int read_raw_photo_method() {
	
	std::cout << "Raw to mat info..." << std::endl;
	// https://blog.csdn.net/zhenzhidemaoyi/article/details/127461686
	
	int imgWidth = 4320;
	int imgHeight = 1088;
	int ret = 0;
	int i = 0, j = 0, k = 0, offset = 0;
	long imgFilesize = 0;
	char imgFileName[300];
	cv::String imgName;

#if SAVE_DATA
	FILE* fp = fopen("fileinfo.csv" ,"a+");
#endif

	cv::String imgPath = "C:\\Users\\s05559\\Documents\\00Work_Plane_Groupby_Data\\202303\\*.raw";

	std::vector<cv::String> imgList;
	glob(imgPath, imgList, true);
	
	int ngNum = 0;

	for (int index = 0; index < imgList.size(); index++) {
		memset(imgFileName ,0 ,sizeof(char) * 300);
		imgName = imgList[index].c_str();
		sprintf(imgFileName ,imgList[index].c_str());

		unsigned char* pTestImgBuf = NULL;
		if (imgPath.rfind(".raw") != imgPath.npos) {
			pTestImgBuf = ReadDataFromRawFile(imgFileName ,&imgFilesize);
			if (pTestImgBuf == NULL) {
				std::cout << "读取图像失败。" << std::endl;
				return -1;
			}

			int iRAWTestFlag = 0, iFuncTimes = 0;
			// RAW格式转cv::Mat格式
			cv::Mat src = cv::Mat(imgHeight ,imgWidth ,CV_8UC1 ,pTestImgBuf);

			cv::imshow("raw photo" ,src);
			cv::waitKey(1000);
		}
		free(pTestImgBuf);
	}

#if SAVEDATA
	fclose(fp);
#endif

	return 0;
}