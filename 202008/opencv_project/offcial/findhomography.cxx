
//Mat homography = R_1to2 + d_inv * tvec_1to2*normal.t();
//Mat homography = R2 * R1.t() + d_inv * (-R2 * R1.t() * tvec1 + tvec2) * normal.t();

// H =  cameraMatrix * (R + t * normal/dot) * cameraMatrix.inv()
void Homography1() {

    std::vector<Point2f> imgInc;
    std::vector<Point3f> objInc;

    FILE* fpw;
    fopen_s(&fpw, "C://Users//s05559//Desktop//img.a", "r");
    for (int i = 0; i < 945; i++) {
        float x, y;
        fscanf_s(fpw, "%f %f", &x, &y);
        imgInc.push_back(Point2f(x,y));
    }
    fclose(fpw);

    FILE* fpw2;
    fopen_s(&fpw2, "C://Users//s05559//Desktop//obj.a", "r");
    for (int i = 0; i < 945; i++) {
        float x, z;
        fscanf_s(fpw, "%f %f", &x, &z);
        objInc.push_back(Point3f(x,0,z));
    }
    fclose(fpw);

    cv::Mat cameraMatrix =(Mat_<double>(3, 3) << 1.35926641e+04, 0., 2.22804810e+03, 0., 2.22419668e+04, 6.59137512e+02,0., 0., 1.);
    cv::Mat distCoeff = (Mat_<double>(1, 5) << -9.13034156e-02, - 2.50199791e-02, 2.11531238e-04 , 1.01283472e-03,- 1.28190117e+01);

    cv::Mat rvec1, tvec1;
    solvePnP(objInc, imgInc,cameraMatrix ,distCoeff ,rvec1 ,tvec1);

    std::cout << "R \n" << rvec1 << std::endl;
    std::cout << "T \n" << tvec1 << std::endl;

    Mat R1,T1;
    Rodrigues(rvec1 ,R1);
    Rodrigues(tvec1, T1);

    //计算单应矩阵
    cv::Mat normal = (cv::Mat_<double>(3,1) << 0 , 0 , 1);
    cv::Mat normal1 = R1 * normal;

    cv::Mat origin(3, 1, CV_64F, cv::Scalar(0));
    cv::Mat origin1 = R1 * origin + tvec1;

    double d_inv1 = 1.0 / normal1.dot(origin1);

    cv::Mat homography_euclidean = R1 + d_inv1 * tvec1 * normal1.t();
    homography_euclidean = cameraMatrix * homography_euclidean * cameraMatrix.inv();

    cv::Mat homography = homography_euclidean / homography_euclidean.at<double>(2 ,2);

    std::cout << R1 << std::endl;
}

void main() {

    std::vector<Point2f> imgInc;
    std::vector<Point3f> objInc;
    std::vector<Point2f> objInc_2f;

    FILE* fpw;
    fopen_s(&fpw, "C://Users//s05559//Desktop//img.a", "r");
    for (int i = 0; i < 945; i++) {
        float x, y;
        fscanf_s(fpw, "%f %f", &x, &y);
        imgInc.push_back(Point2f(x, y));
    }
    fclose(fpw);

    FILE* fpw2;
    fopen_s(&fpw2, "C://Users//s05559//Desktop//obj.a", "r");
    for (int i = 0; i < 945; i++) {
        float x, z;
        fscanf_s(fpw, "%f %f", &x, &z);
        objInc.push_back(Point3f(x, 0, z));
        objInc_2f.push_back(Point2f(x,z));
    }
    fclose(fpw);

    cv::Mat cameraMatrix = (Mat_<double>(3, 3) << 1.35926641e+04, 0., 2.22804810e+03, 0., 2.22419668e+04, 6.59137512e+02, 0., 0., 1.);
    cv::Mat distCoeff = (Mat_<double>(1, 5) << -9.13034156e-02, -2.50199791e-02, 2.11531238e-04, 1.01283472e-03, -1.28190117e+01);

    cv::Mat rvec1, tvec1;
    solvePnP(objInc, imgInc, cameraMatrix, distCoeff, rvec1, tvec1);

    std::cout << "R \n" << rvec1 << std::endl;
    std::cout << "T \n" << tvec1 << std::endl;

    Mat R1;
    Rodrigues(rvec1, R1);
    
    //计算单应矩阵
    cv::Mat foundmentalMat(3 ,3 ,CV_64FC1);
    
    for (int i = 0; i < 3; i++) {
        foundmentalMat.at<double>(i, 0) = R1.at<double>(i,0);
        foundmentalMat.at<double>(i, 1) = R1.at<double>(i, 1);
        foundmentalMat.at<double>(i, 2) = tvec1.at<double>(i ,0);
    }

    cv::Mat homgraphy_euclidean = cameraMatrix * foundmentalMat;
    homgraphy_euclidean = homgraphy_euclidean / homgraphy_euclidean.at<double>(2 ,2);

    std::cout <<"Homography 1: \n " << homgraphy_euclidean << std::endl;

    cv::Mat H = findHomography(objInc_2f ,imgInc);
    std::cout << "Homography 2: \n " << H << std::endl;

}