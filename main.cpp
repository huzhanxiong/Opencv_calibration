#include </usr/include/eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <fstream>

#define PI 3.141592653589793
#define calibration 0
#define pnp 0
#define getData_pic 0
#define img_calibration 1

using namespace cv;
using namespace std;

bool getCodePose(vector<Point2f> codePoints);


int main(int argc, char *argv[])
{
    if(calibration)
    {
        ifstream fin("pic_data.txt"); //图片路径
        ofstream fout("calibration_result.txt");    //结果存储

        cout << "开始提取角点......" << endl;
        
        int image_count = 0;
        Size image_size;
        Size board_size = Size(7, 9);   //标定版上行列角点数
        vector<Point2f> image_points_buf; //缓存每幅图像上检测到的角点
        vector< vector<Point2f> > image_points_seq; //保存检测的所有角点
        string filename;
        //int count = -1;

        while( getline(fin, filename) )
        {
            image_count++;
            cout << "image_count = " << image_count << endl;
            // cout << "-->count = " << count;
            Mat imageInput = imread(filename);
            // cout << imageInput.channels() << endl;
            // imshow(filename, imageInput);
            // waitKey(0);
            if(image_count == 1)
            {
                image_size.width = imageInput.cols;
                image_size.height = imageInput.rows;
                cout << "image_size.width = " << image_size.width << endl;
                cout << "iamge_size.height = " << image_size.height << endl;
            }

            //提取角点
            if(findChessboardCorners(imageInput, board_size, image_points_buf) == 0)
            {
                cout << "can not find chessboard corners! \n";
                exit(1);
            }
            else
            {
                Mat view_gray;
                cvtColor(imageInput, view_gray, CV_RGB2GRAY);
                //亚像素精确化
                find4QuadCornerSubpix(view_gray, image_points_buf, Size(11,11));
                image_points_seq.push_back(image_points_buf);
                //显示角点位置
                drawChessboardCorners(view_gray, board_size, image_points_buf, true);
                imshow("camera_calibration", view_gray);
                waitKey(100);
            }
        }

        int total = image_points_seq.size();
        cout << "total = " << total << endl;
        int CornerNum = board_size.width * board_size.height;   //每张图片上总的角点数
        for(int n=0; n<total; n++)
        {
            if(n % CornerNum == 0)
            {
                int i = -1;
                i = n/CornerNum;
                int j = i+1;
                cout << "--> 第 " << j << "图片数据 --> : " << endl;
            }
            if(n % 3 == 0)
            {
                cout << endl;
            }
            else
            {
                cout.width(10);
            }

            //输出所有角点
            cout << " -->" << image_points_seq[n][0].x;
            cout << " -->" << image_points_seq[n][0].y;
        }
        cout << "\n 角点提取完成！ \n";

        cout << "开始标定......";

        Size square_size = Size(23, 23);    //实际测量得到的标定板上每个盘格的大小 mm
        std::vector< std::vector<Point3f> > object_points;  //保存标定版上角点的三维坐标

        Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); //相机内参矩阵
        std::vector<int> point_counts;  //每幅图像中角点的数量
        Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));   //相机的5个畸变系数: k1,k2,p1,p2,k3
        std::vector<Mat> tvecsMat;  //每幅图像的旋转向量
        std::vector<Mat> rvecsMat;  //每幅图像的平移向量
        //初始化标定板上角点的三维坐标
        for(int t=0; t<image_count; t++)
        {
            std::vector<Point3f> tempPointSet;
            for(int i=0; i<board_size.height; i++)
            {
                for(int j=0; j<board_size.width; j++)
                {
                    Point3f realPoint;
                    //假设标定板放在世界坐标系中z=0的平面上
                    realPoint.x = i*square_size.width;
                    realPoint.y = j*square_size.height;
                    realPoint.z = 0;
                    tempPointSet.push_back(realPoint);
                }
            }
            object_points.push_back(tempPointSet);
        }
        //初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板
        for(int i=0; i<image_count; i++)
        {
            point_counts.push_back(board_size.width * board_size.height);
        }

        calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
        cout << "标定完成! \n";

        cout << "开始评价标定结果......\n";
        double total_err = 0.0; //所有图像的平均误差总和
        double err = 0.0;   //每幅图像的平均误差
        std::vector<Point2f> image_points2; //保存重新计算得到的投影点
        cout << "\t每幅图像的标定误差: \n";
        fout << "每幅图像的标定误差: \n";
        for(int i=0; i<image_count; i++)
        {
            std::vector<Point3f> tempPointSet = object_points[i];
            //通过得到的相机内外参数，对空间的三维点进行重新投影计算，得到新的投影点
            projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
            //计算新的投影点和旧的投影点之间的误差
            std::vector<Point2f> tempImagePoint = image_points_seq[i];
            Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
            Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
            for(int j=0; j<tempImagePoint.size(); j++)
            {
                image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
                tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
            }

            err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
            total_err += err/= point_counts[i];
            cout << "第" << i+1 << "幅图像的平均误差: " << err << "像素" << endl;
            fout << "第" << i+1 << "幅图像的平均误差: " << err << "像素" << endl;
        }
        cout << "总体平均误差: " << total_err/image_count << "像素" << endl;
        fout << "总体平均误差: " << total_err/image_count << "像素" << endl;
        cout << "评价完成！ " << endl;

        cout << "开始保存标定结果......" << endl;
        Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
        fout << "相机内参数矩阵: " << endl;
        fout << cameraMatrix << endl << endl;
        fout << "畸变系数: \n";
        fout << distCoeffs << endl <<endl;
        for(int i=0; i<image_count; i++)
        {
            fout << "第" << i+1 << "幅图像的旋转向量: " << endl;
            fout << tvecsMat[i] << endl;

            Rodrigues(tvecsMat[i], rotation_matrix);
            fout << "第" << i+1 << "幅图像的旋转矩阵: " << endl;
            fout << rotation_matrix << endl;
            fout << "第" << i+1 << "幅图像的平移向量: " << endl;
            fout << rvecsMat[i] << endl << endl;
        }
        cout << "完成保存" << endl;
        fout << endl;
        waitKey(0);
        return 0;
    }

    if(pnp)
    {
        vector<Point3f> world_coordinate;
        Mat world_coordinate_mat;
        world_coordinate.clear();
        world_coordinate.push_back(Point3f(50, 50, 0)); //2730   //单位mm
        world_coordinate.push_back(Point3f(-50, 50, 0));
        world_coordinate.push_back(Point3f(-50, -50, 0));
        world_coordinate.push_back(Point3f(50, -50, 0));
        Mat(world_coordinate).convertTo(world_coordinate_mat, CV_32F);
    
        vector<Point2f> image_points;
        image_points.clear();
        image_points.push_back(Point2f(571, 456));
        image_points.push_back(Point2f(555, 488));
        image_points.push_back(Point2f(523, 472));
        image_points.push_back(Point2f(538, 440));

        // double camD[9] = {  };
        // double distCoeffD[5] = {   };
        // Mat camera_matrix = Mat(3, 3, CV_32FC1, camD); 
        // Mat distortion_coefficients = Mat(1, 5, CV_32FC1, distCoeffD);   
    
    
		Mat camMatrix; //相机内参矩阵
		Mat distCoeff; //相机畸变系数
		camMatrix = (Mat_<double>(3, 3) << 1503.24441, 0, 316.70629, 0, 1526.24740, 261.94298, 0, 0, 1);                                                                         
		distCoeff  = (Mat_<double>(5, 1) << -0.05103, 6.50903, -0.01812,  -0.08288, -13.22825);                        

        Mat Rvec;
        Mat_<float> Tvec;
        Mat raux, taux;
        solvePnP(world_coordinate_mat, image_points, camMatrix, distCoeff, raux, taux, false, CV_ITERATIVE);

        cout << raux << endl << taux << endl;

        raux.convertTo(Rvec, CV_32F);   //旋转向量
        taux.convertTo(Tvec, CV_32F);   //平移向量

        Mat_<float> rotMat(3, 3);
        Rodrigues(Rvec, rotMat);  //使用罗德里格斯变换变成旋转矩阵

        //格式转换
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> R_n;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> T_n;
        cv2eigen(rotMat, R_n);
        cv2eigen(Tvec, T_n);
        Eigen::Vector3f P_oc;
        P_oc = -R_n.inverse() * T_n;
        printf("x:%.1f  y:%.1f  z:%.1f\n", P_oc[0]/10, P_oc[1]/10, P_oc[2]/10);

        float theta_x, theta_y, theta_z;
        theta_x = atan2(rotMat[2][1], rotMat[2][2]) * 57.2958;
        theta_y = atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2])) * 57.2958; 
        theta_z = atan2(rotMat[1][0], rotMat[0][0]) * 57.2958;
        cout << theta_x << " " << theta_y << " " << theta_z << endl;
    }

	if(getData_pic)
	{
		vector<Point2f> image_points;
		image_points.clear();
		image_points.push_back(Point2f(314, 250));
		image_points.push_back(Point2f(314, 213));
		image_points.push_back(Point2f(351, 213));
		image_points.push_back(Point2f(351, 250));
		getCodePose(image_points);
	}
	
	if(img_calibration)
	{
		/*VideoCapture inputVideo(0);
		if (!inputVideo.isOpened())
		{
			cout << "Could not open the input video: " << endl;
			return -1;
		}*/
		Mat frame;
		Mat frameCalibration;

		//inputVideo >> frame;
		frame = imread("./image2/9.png");
		
		Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
		cameraMatrix.at<double>(0, 0) = 1195.3866;
		cameraMatrix.at<double>(0, 2) = 324.5385;
		cameraMatrix.at<double>(1, 1) = 1309.0698;
		cameraMatrix.at<double>(1, 2) = 214.0473;
		cameraMatrix.at<double>(2, 2) = 1;

		Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
		distCoeffs.at<double>(0, 0) = -2.2617;
		distCoeffs.at<double>(1, 0) = 13.9632;
		distCoeffs.at<double>(2, 0) = 0.0084;
		distCoeffs.at<double>(3, 0) = 0.0600;
		distCoeffs.at<double>(4, 0) = -56.3373;

		Mat map1, map2;
		//去畸变并保留最大图
		/***initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, frame.size(), 1, frame.size(), 0),
		frame.size(), CV_16SC2, map1, map2);**/
		// 去畸变至全图
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), cameraMatrix, frame.size(), CV_16SC2, map1, map2);
/**FileStorage fs1("map1.xml", FileStorage::WRITE);
cv::write(fs1, "map1_data", map1);
fs1.release();
FileStorage fs2("map2.xml", FileStorage::WRITE);
cv::write(fs2, "map2_data", map2);
fs2.release();**/
/**FileStorage fs("map1.xml", FileStorage::READ);
read(fs["map1_data"], Mat);
fs.release();**/
		frame = imread("./image2/1.png");            
		remap(frame, frameCalibration, map1, map2, INTER_LINEAR);
		
		imshow("Origianl", frame);
		imshow("Calibration", frameCalibration);
		char key = waitKey(0);
		if (key == 27)
			return 0;
	}
}





bool getCodePose(vector<Point2f> codePoints)
{
    float hViewAngle = 0.927 * 3 / 4;

    //获得四个点的坐标
    double x0 = codePoints[0].x;
    double y0 = codePoints[0].y;
    double x1 = codePoints[1].x;
    double y1 = codePoints[1].y;
    double x2 = codePoints[2].x;
    double y2 = codePoints[2].y;
    double x3 = codePoints[3].x;
    double y3 = codePoints[3].y;
    //左边沿纵向差
    double leftH=y1-y0;
    //右边沿纵向差
    double rightH=y2-y3;
    //必须保证0点高于1点，3点高于2点
//    if(leftH<0||rightH<0)
//        return false;
    //左边沿横向差
    double leftW=abs(x0-x1);
    //右边沿横向差
    double rightW=abs(x2-x3);
    //不能太倾斜
   if(max(leftW/leftH,rightW/rightH) > 0.1)
       return false;
    //上下视角一半的正切值，因为一直要用，所以先计算出来
    double tanHalfView=tan(hViewAngle/2);
    double leftLen=sqrt(leftH*leftH+leftW*leftW);
    double rightLen=sqrt(rightH*rightH+rightW*rightW);
    //左边沿的深度
    int Frame_height = 600;
    float qrSize = 5.0; //cm
    double leftZ=Frame_height*qrSize/tanHalfView/2/leftLen;
    //右边沿的深度
    double rightZ=Frame_height*qrSize/tanHalfView/2/rightLen;
    //得到中心点的深度
    double z=(leftZ+rightZ)/2;
    //计算b的正弦值
    double sinB=(leftZ-rightZ)/qrSize;
    printf("leftZ = %.2f   rightZ = %.2f  sinb = %.2f\n", leftZ, rightZ, sinB);
    if(sinB>1)
        return false;
    //得到b
    double b=asin(sinB);
    cout << "angle b=" << b*180/3.14 << endl;

    // //两条对角线的系数和偏移
    // double k1=(y2-y0)/(x2-x0);
    // double b1=(x2*y0-x0*y2)/(x2-x0);
    // double k2=(y3-y1)/(x3-x1);
    // double b2=(x3*y1-x1*y3)/(x3-x1);
    // //两条对角线交点的X坐标
    // double crossX=-(b1-b2)/(k1-k2);
    // //计算a的正切值
    // int Frame_width = 600;
    // double tanA=tanHalfView*(2*crossX-Frame_width)/Frame_height;
    // //得到a
    // double a=atan(tanA);
     
    // printf("a=%.2f  b=%.2f  z=%.2f\n", a, b, z);
    // return true;
}
