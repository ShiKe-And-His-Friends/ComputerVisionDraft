#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
using namespace std;
using namespace cv;
namespace
{
    enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
    void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
    {
        corners.resize(0);
        switch (patternType)
        {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    corners.push_back(Point3f(float(j * squareSize),
                        float(i * squareSize), 0));
            break;
        case ASYMMETRIC_CIRCLES_GRID:
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
                        float(i * squareSize), 0));
            break;
        default:
            CV_Error(Error::StsBadArg, "Unknown pattern type\n");
        }
    }
    Mat computeHomography(const Mat& R_1to2, const Mat& tvec_1to2, const double d_inv, const Mat& normal)
    {
        Mat homography = R_1to2 + d_inv * tvec_1to2 * normal.t();
        return homography;
    }
    Mat computeHomography(const Mat& R1, const Mat& tvec1, const Mat& R2, const Mat& tvec2,
        const double d_inv, const Mat& normal)
    {
        Mat homography = R2 * R1.t() + d_inv * (-R2 * R1.t() * tvec1 + tvec2) * normal.t();
        return homography;
    }
    void computeC2MC1(const Mat& R1, const Mat& tvec1, const Mat& R2, const Mat& tvec2,
        Mat& R_1to2, Mat& tvec_1to2)
    {
        //c2Mc1 = c2Mo * oMc1 = c2Mo * c1Mo.inv()
        R_1to2 = R2 * R1.t();
        tvec_1to2 = R2 * (-R1.t() * tvec1) + tvec2;
    }
    void homographyFromCameraDisplacement(const string& img1Path, const string& img2Path, const Size& patternSize,
        const float squareSize, const string& intrinsicsPath)
    {
        Mat img1 = imread(samples::findFile(img1Path));
        Mat img2 = imread(samples::findFile(img2Path));

        // my cheerboard photo
        // Mat img1 = imread("E:\\computerVersion\\ComputerVisionDraft\\202204\\OPENCV_CODE\\20220629_CalibrateCamera01\\CameraData\\Img-01.bmp");
        // Mat img2 = imread("E:\\computerVersion\\ComputerVisionDraft\\202204\\OPENCV_CODE\\20220629_CalibrateCamera01\\CameraData\\Img-08.bmp");

        vector<Point2f> corners1, corners2;
        
        
        bool found1 = findChessboardCorners(img1, patternSize, corners1);
        bool found2 = findChessboardCorners(img2, patternSize, corners2);
        if (!found1 || !found2)
        {
            cout << "Error, cannot find the chessboard corners in both images." << endl;
            return;
        }
        vector<Point3f> objectPoints;
        calcChessboardCorners(patternSize, squareSize, objectPoints);
        FileStorage fs(samples::findFile(intrinsicsPath), FileStorage::READ);
        Mat cameraMatrix, distCoeffs;
        fs["camera_matrix"] >> cameraMatrix;
        fs["distortion_coefficients"] >> distCoeffs;
        Mat rvec1, tvec1;
        solvePnP(objectPoints, corners1, cameraMatrix, distCoeffs, rvec1, tvec1);
        Mat rvec2, tvec2;
        solvePnP(objectPoints, corners2, cameraMatrix, distCoeffs, rvec2, tvec2);
        Mat img1_copy_pose = img1.clone(), img2_copy_pose = img2.clone();
        Mat img_draw_poses;
        drawFrameAxes(img1_copy_pose, cameraMatrix, distCoeffs, rvec1, tvec1, 2 * squareSize);
        drawFrameAxes(img2_copy_pose, cameraMatrix, distCoeffs, rvec2, tvec2, 2 * squareSize);
        hconcat(img1_copy_pose, img2_copy_pose, img_draw_poses);
        imshow("Chessboard poses", img_draw_poses);
        Mat R1, R2;
        Rodrigues(rvec1, R1);
        Rodrigues(rvec2, R2);
        Mat R_1to2, t_1to2;
        computeC2MC1(R1, tvec1, R2, tvec2, R_1to2, t_1to2);
        Mat rvec_1to2;
        Rodrigues(R_1to2, rvec_1to2);
        Mat normal = (Mat_<double>(3, 1) << 0, 0, 1);
        Mat normal1 = R1 * normal;
        Mat origin(3, 1, CV_64F, Scalar(0));
        Mat origin1 = R1 * origin + tvec1;
        double d_inv1 = 1.0 / normal1.dot(origin1);
        Mat homography_euclidean = computeHomography(R_1to2, t_1to2, d_inv1, normal1);
        Mat homography = cameraMatrix * homography_euclidean * cameraMatrix.inv();
        homography /= homography.at<double>(2, 2);
        homography_euclidean /= homography_euclidean.at<double>(2, 2);
        //Same but using absolute camera poses instead of camera displacement, just for check
        Mat homography_euclidean2 = computeHomography(R1, tvec1, R2, tvec2, d_inv1, normal1);
        Mat homography2 = cameraMatrix * homography_euclidean2 * cameraMatrix.inv();
        homography_euclidean2 /= homography_euclidean2.at<double>(2, 2);
        homography2 /= homography2.at<double>(2, 2);
        cout << "\nEuclidean Homography:\n" << homography_euclidean << endl;
        cout << "Euclidean Homography 2:\n" << homography_euclidean2 << endl << endl;
        Mat H = findHomography(corners1, corners2);
        cout << "\nfindHomography H:\n" << H << endl;
        cout << "homography from camera displacement:\n" << homography << endl;
        cout << "homography from absolute camera poses:\n" << homography2 << endl << endl;
        Mat img1_warp;
        warpPerspective(img1, img1_warp, H, img1.size());
        Mat img1_warp_custom;
        warpPerspective(img1, img1_warp_custom, homography, img1.size());
        imshow("Warped image using homography computed from camera displacement", img1_warp_custom);
        Mat img_draw_compare;
        hconcat(img1_warp, img1_warp_custom, img_draw_compare);
        imshow("Warped images comparison", img_draw_compare);
        Mat img1_warp_custom2;
        warpPerspective(img1, img1_warp_custom2, homography2, img1.size());
        imshow("Warped image using homography computed from absolute camera poses", img1_warp_custom2);
        waitKey();
    }
    const char* params
        = "{ help h         |       | print usage }"
        "{ image1         | left02.jpg | path to the source chessboard image }"
        "{ image2         | left01.jpg | path to the desired chessboard image }"
        "{ intrinsics     | left_intrinsics.yml | path to camera intrinsics }"
        "{ width bw       | 9     | chessboard width }"
        "{ height bh      | 6     | chessboard height }"
        "{ square_size    | 0.025 | chessboard square size }";
}
int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);
    if (parser.has("help"))
    {
        parser.about("Code for homography tutorial.\n"
            "Example 3: homography from the camera displacement.\n");
        parser.printMessage();
        return 0;
    }
    Size patternSize(parser.get<int>("width"), parser.get<int>("height"));
    float squareSize = (float)parser.get<double>("square_size");
    homographyFromCameraDisplacement(parser.get<String>("image1"),
        parser.get<String>("image2"),
        patternSize, squareSize,
        parser.get<String>("intrinsics"));
    return 0;
}