#include "ClibaHelp.h"

// 棋盘格坐标轴的描点
void ClibaHelp::calcChessboards(const Size &chessboardSize, vector<Point3f> &corners) {
	corners.resize(0);
	for (int i = 0; i < chessboardSize.width; i++) {
		for (int j = 0; j < chessboardSize.height; j++) {
			float x = (float)(i*square_size);
			float y = (float)(j*square_size);
			corners.push_back(
				Point3f(x, y, 0)
			);
		}
	}
	vector<Point3f>::iterator it, end;
	it = corners.begin();
	end = corners.end();
	cout << "Chessboard aixs number :" << (end - it) << endl;

	for (; it != end; it++) {
		cout << "Chessboard aixs x" << (*it).x << " y " << (*it).y << endl;
	}
}

// 分解矩阵
void ClibaHelp::decomposeMatrix(const Mat &mat1, const Mat &mat2, Size &patternSize
	, vector<Point3f> &axis, vector<Point2f> &corners1, vector<Point2f> &corners2) {
	String intrinsicsPath = samples::findFile("left_intrinsics.yml");
	cout << "Camera samples intrinsics path: " << intrinsicsPath << endl;
	FileStorage file(intrinsicsPath, FileStorage::READ);

	// 读取示例的相机参数
	Mat cameraIntrinsicsMatrix, cameraDistCoffes;
	file["camera_matrix"] >> cameraIntrinsicsMatrix;
	file["distortion_coefficients"] >> cameraDistCoffes;
	file.release();
	cout << "Intrinsics " << endl << cameraIntrinsicsMatrix << endl;
	cout << "Dist Coffes" << endl << cameraDistCoffes << endl;

	// 用已知点求空间点的P3P计算
	Mat rVecs_1, tVecs_1;
	solvePnP(axis, corners1, cameraIntrinsicsMatrix, cameraDistCoffes, rVecs_1, tVecs_1);
	cout << "rVecs_1 " << endl << rVecs_1 << endl;
	cout << "tVecs_1" << endl << tVecs_1 << endl;

	Mat rVecs_2, tVecs_2;
	solvePnP(axis, corners2, cameraIntrinsicsMatrix, cameraDistCoffes, rVecs_2, tVecs_2);
	cout << "rVecs_2 " << endl << rVecs_2 << endl;
	cout << "tVecs_2" << endl << tVecs_2 << endl;

	// 求旋转矩阵R1 R1
	Mat R1, R2, T1, T2;
	Rodrigues(rVecs_1, R1);
	Rodrigues(rVecs_2, R2);

	// 求转换的矩阵
	Mat R1_to_R2, t1_to_t2;
	R1_to_R2 = R2 * R1.t();
	t1_to_t2 = R2 * (-R1.t()* tVecs_1) + tVecs_2;
	cout << "R1_to_R2: " << endl << R1_to_R2 << endl;
	cout << "t1_to_t2: " << endl << t1_to_t2 << endl;
	Mat rvec1_to_rvec2;
	Rodrigues(R1_to_R2, rvec1_to_rvec2);

	// camera1 cmaera2 pose estimated
	//

	// planer n vector
	Mat normal = (Mat_<double>(3, 1) << 0, 0, 1);
	Mat normal1 = R1 * normal;

	//distance d  between camera and planer
	Mat origin(3, 1, CV_64F, Scalar(0));
	Mat origin1 = R1 * origin + tVecs_1;
	double d_inv1 = 1.0 / normal1.dot(origin1);

	// 计算单应矩阵
	Mat homography_euclidean = R1_to_R2 + d_inv1 * t1_to_t2 * normal1.t();
	cout << "homography euclidean (no normal) :" << endl << homography_euclidean << endl;
	Mat homograph = cameraIntrinsicsMatrix * homography_euclidean * cameraIntrinsicsMatrix.inv();
	cout << "homography (no normal) :" << endl << homograph << endl;
	homography_euclidean /= homography_euclidean.at<double>(2, 2);
	homograph /= homograph.at<double>(2, 2);
	cout << "homography euclidean :" << endl << homography_euclidean << endl;
	cout << "homography :" << endl << homograph << endl;

	//对比结果
	Mat H_sample_1 = findHomography(corners1, corners2);
	vector<Mat> r_decomp, t_decomp, normal_decomp;
	int solutions = decomposeHomographyMat(H_sample_1, cameraIntrinsicsMatrix, r_decomp, t_decomp, normal_decomp);
	cout << "Decompse homography matrix estimated by findHomograpy() using (photo1,photo2):" << endl;
	homographyInfoClean();
	for (int i = 0; i < solutions; i++) {
		double factor_d1 = 1.0 / d_inv1;
		Mat r_mat_decompse;
		Rodrigues(r_decomp[i], r_mat_decompse);
		cout << "soulution " << i << " :" << endl;
		cout << "rvec from homography decomposition:" << r_mat_decompse.t() << endl;
		cout << "rvec from camera displacement:" << rvec1_to_rvec2.t() << endl;
		cout << "tvec from homography decomposition:" << t_decomp[i].t() << " and scale by d:" << factor_d1 * t_decomp[i].t() << endl;
		cout << "tvec from camera displacement:" << t1_to_t2.t() << endl;
		cout << "plane normal from homography decomposition" << normal_decomp[i].t() << endl;
		cout << "plane normal at camera 1 pose;" << normal1.t() << endl;
		homographyInfoSave(r_decomp[i], t_decomp[i], normal_decomp[i], i);
	}

	solutions = decomposeHomographyMat(homograph, cameraIntrinsicsMatrix, r_decomp, t_decomp, normal_decomp);
	cout << "Decompse homography matrix computed from the camera displacement:" << endl;
	for (int i = 0; i < solutions; i++) {
		double factor_d1 = 1.0 / d_inv1;
		Mat r_mat_decompse;
		Rodrigues(r_decomp[i], r_mat_decompse);
		cout << "soulution " << i << " :" << endl;
		cout << "rvec from homography decomposition:" << r_mat_decompse.t() << endl;
		cout << "rvec from camera displacement:" << rvec1_to_rvec2.t() << endl;
		cout << "tvec from homography decomposition:" << t_decomp[i].t() << " and scale by d:" << factor_d1 * t_decomp[i].t() << endl;
		cout << "tvec from camera displacement:" << t1_to_t2.t() << endl;
		cout << "plane normal from homography decomposition" << normal_decomp[i].t() << endl;
		cout << "plane normal at camera 1 pose;" << normal1.t() << endl;
		homographyInfoSave(r_decomp[i], t_decomp[i], normal_decomp[i] ,i);
	}

}

// yaml文件删除
void  ClibaHelp::homographyInfoClean() {
	FileStorage file(homograph_yaml_dir, FileStorage::WRITE);
	file << "Author " << "sk95120";
	file << "Time " << "NONE";
	file.release();
}

// 单应矩阵的信息存yaml文件
void  ClibaHelp::homographyInfoSave(const Mat &r, const Mat &t, const Mat& n ,int num){
	FileStorage file(homograph_yaml_dir ,FileStorage::APPEND);
	file << "Num " << num;
	file << "Rotate " << r;
	file << "Transoform " << t;
	file << "Normalize " << n;

	file.release();
}