// todo debug
	if (found1 || found2) {
		Mat gray;
		cvtColor(photo1 , photo1 ,COLOR_RGB2GRAY);
		cvtColor(photo2 , photo2, COLOR_RGB2GRAY);
		find4QuadCornerSubpix(photo1, photo_corner_1 , patternSize);
		find4QuadCornerSubpix(photo2, photo_corner_2 , patternSize);
		drawChessboardCorners(photo1 ,patternSize , photo_corner_1 ,true);
		drawChessboardCorners(photo2, patternSize, photo_corner_2, true);
		imshow("quad corner1" ,photo1);
		imshow("quad corner2", photo2);
		waitKey(1000);
		destroyAllWindows();
	}