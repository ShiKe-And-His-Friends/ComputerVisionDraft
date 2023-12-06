//检测米粒
int main() {

    //加载图片
    cv::Mat rice_image = cv::imread("C://Users//s05559//Pictures//rice.tif" ,cv::IMREAD_UNCHANGED);
    
    //高斯滤波
    cv::Mat rice_image_gaussian;
    cv::GaussianBlur(rice_image , rice_image_gaussian,cv::Size(5,5) ,0);

    //腐蚀减
    cv::Mat rice_erode_image ,rice_dilation_image;
    cv::erode(rice_image_gaussian ,rice_erode_image ,cv::Mat() ,cv::Point(-1,-1) ,5 ,cv::BORDER_DEFAULT);
    cv::dilate(rice_erode_image,rice_dilation_image ,cv::Mat() ,cv::Point(-1,-1) ,5 ,cv::BORDER_DEFAULT);
    cv::Mat rice_sub_image;
    cv::subtract(rice_image ,rice_erode_image ,rice_sub_image);


    //二值化
    cv::Mat rice_image_binary;
    cv::threshold(rice_sub_image, rice_image_binary, 50, 255, cv::THRESH_BINARY);

    //边缘检测
    //cv::Mat rice_image_canny;
    //cv::Canny(rice_sub_image ,rice_image_canny ,0,255 ,3);

    //找轮廓
    std::vector<std::vector<cv::Point>> contourss;
    cv::findContours(rice_image_binary,contourss ,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);
    std::cout << "轮廓点数量 " << contourss.size() << std::endl;

    cv::Mat rice_find_image;
    cv::cvtColor(rice_image_binary,rice_find_image ,cv::COLOR_GRAY2BGR);
    cv::drawContours(rice_find_image ,contourss ,-1 ,cv::Scalar(255 ,0 ,0) );
    cv::imwrite("C://Users//s05559//Pictures//rice_temp.tif" , rice_find_image);

    cv::namedWindow("rice_windows" ,cv::WINDOW_NORMAL);
    cv::imshow("rice_windows" , rice_find_image);
    cv::waitKey(1000);

    return 0;
}