#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

#define WINDOW_NAME "【程序窗口】" 
#define Patch "E:\\TestDate\\hudeilan\\Date\\校正后的图像.jpg"
//E:\\TestDate\\verify\\01.jpg   用于验证正确性
//-----------------------------------【全局变量声明部分】-------------------------------------------
//		描述：全局变量的声明
Point g_Point;
vector<Point2f> points(4);	//储存点的坐标，后期看看是否可以优化
bool g_bPoint = false;//是否进行绘制
RNG g_rng(12345);
int cnt = 0;//初始化调用次数
Mat transformed;	//存储透视变换处理后的图片
double RectA4 = 100.0;
//-----------------------------------------------------------------------------------------------
int g_nThresholdValue = 100;
int g_nThresholdType = 3;
Mat g_dstImage;
double e = 35;//设置ROI距离顶点的边界为多少
//-----------------------------------------------------------------------------------------------

void GetCoordinates(Mat srcImage);
void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawFilledCircle(cv::Mat& img, Point center);

void tf(Point2f q1, Point2f q2, Point2f q3, Point2f q4);
void HSVSplit(Mat image);
void GoodFeaturesToTrack(Mat g_srcImage, double minDistance);
double ContourArea(Point2f q1, Point2f q2, Point2f q3, Point2f q4);
void on_Threshold(int, void*);//回调函数
int Threshold(Mat g_srcImage);//阈值化
int OpenOperation(Mat image, MorphTypes a);
int K_Means(Mat pic);//K-Means算法进行分割
double Contours(Mat src);
double RectArea(Mat src,Point2f q1, Point2f q2, Point2f q3, Point2f q4);
bool cmpy(cv::Point const& a, cv::Point const& b);//对点的yx值进行从小到大的排序
bool cmpx(cv::Point const& a, cv::Point const& b);
int RectMask(Mat image, double e);
int main() {
	transformed = imread(Patch);
	GetCoordinates(transformed);
 	waitKey(0);
	destroyWindow(WINDOW_NAME);
	//透视变换
	tf(points[0], points[1], points[2], points[3]);
	waitKey(0);
	//转换HSV色度空间进行分割
	HSVSplit(transformed);//K-Means算法更好
	//K_Means(transformed);


	//确定四点的坐标
	double minDistance = 350;//角点之间的最小距离
	GoodFeaturesToTrack(transformed, minDistance);
	//OpenOperation(transformed, MORPH_OPEN);//不需要阈值化了
	//OpenOperation(transformed, MORPH_CLOSE);//消除矩形

	sort(points.begin(), points.end(), cmpy);//对点的y值进行从小到大的排序,来确定左上角和右下角的坐标
	cmpx(points[0],points[1]);//对排序y值的前两个进行x值的排序
	cmpx(points[2], points[3]);//对排序y值的后两个进行x值的排序
	//计算矩形的面积
	double R = RectArea(transformed, points[0], points[1], points[2], points[3]);
	//计算树叶的面积
	double L = Contours(transformed);
	double area = RectA4 * L / R;
	cout << "树叶的真实面积为" << area << endl;
	//Contours(transformed);
	//阈值化
	//Threshold(transformed);
	//GoodFeaturesToTrack(transformed);
	waitKey(0);
	
	//计算矩形的面积


}

void PerspectiveTransform() {
	cout << "进行自动透视变换" << endl;

}
//-----------------------------------------------------------------------------------------------
//GetCoordinates();使用的函数
void GetCoordinates(Mat srcImage) {
	Mat tempImage;

	srcImage.copyTo(tempImage);

	//【2】设置鼠标操作回调函数
	namedWindow(WINDOW_NAME);
	setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);

	while (1)
	{
		srcImage.copyTo(tempImage);//拷贝源图到临时变量
		if (g_bPoint) DrawFilledCircle(tempImage, g_Point);//当进行绘制的标识符为真，则进行绘制
		imshow(WINDOW_NAME, tempImage);
		if (waitKey(10) == 27 || cnt > 3) break;//按下ESC键，或者点四次鼠标则程序退出
	}
}
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{

	Mat& image = *(cv::Mat*) param;
	switch (event)
	{
		//左键按下消息
	case EVENT_LBUTTONDOWN:
	{
		g_bPoint = true;
		g_Point = Point2f(x, y);//记录起始点
	}
	break;

	//左键抬起消息
	case EVENT_LBUTTONUP:
	{
		g_bPoint = false;//置标识符为false
		//对宽和高小于0的处理
		++cnt;//获取函数调用的次数
		DrawFilledCircle(image, g_Point);
		cout << "第" << cnt << "个点 " << "坐标为：" << g_Point << endl;
		int i = cnt - 1;
		points[i] = g_Point;

		//下面注释代码用来测试points是否会溢出
		//cout << points << endl;
	}
	break;
	}
}
void DrawFilledCircle(cv::Mat& img, Point center)
{
	int thickness = -1;
	int lineType = 8;
	circle(img, center, 10, Scalar(0, 0, 255), thickness, lineType);

}
//-----------------------------------------------------------------------------------------------
//进行透视变换
void tf(Point2f q1, Point2f q2, Point2f q3, Point2f q4) {
	Mat src = imread(Patch);
	Point Q1 = q1;
	Point Q2 = q2;
	Point Q3 = q3;
	Point Q4 = q4;

	//调试代码
	//Point Q1 = Point2f(322, 242);
	//Point Q2 = Point2f(638, 348);
	//Point Q3 = Point2f(539, 530);
	//Point Q4 = Point2f(188, 375);

	// compute the size of the card by keeping aspect ratio.
	double ratio = 1.0;//比例系数因为改程序模板为 正方形 ，此值设置为1.0
	double cardH = sqrt((Q3.x - Q2.x)*(Q3.x - Q2.x) + (Q3.y - Q2.y)*(Q3.y - Q2.y));//或者你可以给出自己的身高
	double cardW = ratio * cardH;
	Rect R(Q1.x, Q1.y, cardW, cardH);

	Point R1 = Point2f(R.x, R.y);
	Point R2 = Point2f(R.x + R.width, R.y);
	Point R3 = Point2f(Point2f(R.x + R.width, R.y + R.height));
	Point R4 = Point2f(Point2f(R.x, R.y + R.height));

	std::vector<Point2f> quad_pts;
	std::vector<Point2f> squre_pts;

	quad_pts.push_back(Q1);
	quad_pts.push_back(Q2);
	quad_pts.push_back(Q3);
	quad_pts.push_back(Q4);

	squre_pts.push_back(R1);
	squre_pts.push_back(R2);
	squre_pts.push_back(R3);
	squre_pts.push_back(R4);


	Mat transmtx = getPerspectiveTransform(quad_pts, squre_pts);
	int offsetSize = 150;
	transformed = Mat::zeros(R.height + offsetSize, R.width + offsetSize, CV_8UC3);
	warpPerspective(src, transformed, transmtx, src.size(),1,1);//原是transmtx.size()

	//rectangle(src, R, Scalar(0,255,0),1,8,0);

	line(src, Q1, Q2, Scalar(0, 0, 255), 1, LINE_AA, 0);
	line(src, Q2, Q3, Scalar(0, 0, 255), 1, LINE_AA, 0);
	line(src, Q3, Q4, Scalar(0, 0, 255), 1, LINE_AA, 0);
	line(src, Q4, Q1, Scalar(0, 0, 255), 1, LINE_AA, 0);

	imshow("quadrilateral", transformed);
	imshow("src", src);
	waitKey();
	destroyWindow("quadrilateral");
	destroyWindow("src");
}
//-----------------------------------------------------------------------------------------------
void HSVSplit(Mat image) {
	Mat hsvimage, hue;
	imshow("image", image);
	cvtColor(image, hsvimage,COLOR_BGR2HSV); //RGB到HSV颜色空间的转换
	imshow("HSV", hsvimage); //直接把HSV规格的图像以RGB格式显示。显示出来的图像会与原图不同
	vector<Mat> hsv;
	split(hsvimage, hsv);//将HSV三个通道分离
	imshow("S色度空间", hsv.at(1));
	transformed = hsv.at(1);
	waitKey();
	destroyWindow("image");
	destroyWindow("HSV");
	destroyWindow("S色度空间");
}
void GoodFeaturesToTrack(Mat g_srcImage,double minDistance) {
	Mat g_grayImage = g_srcImage.clone();
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	int g_maxCornerNumber = 4;
	//【1】对变量小于等于1时的处理
	if (g_maxCornerNumber <= 1) { g_maxCornerNumber = 1; }

	//【2】Shi-Tomasi算法（goodFeaturesToTrack函数）的参数准备
	vector<Point2f> corners;
	double qualityLevel = 0.01;//角点检测可接受的最小特征值

	int blockSize = 3;//计算导数自相关矩阵时指定的邻域范围
	double k = 0.04;//权重系数
	Mat copy = g_srcImage.clone();	//复制源图像到一个临时变量中，作为感兴趣区域

	//写入ROI区域	
	RectMask(g_grayImage, e);
	//【3】进行Shi-Tomasi角点检测
	goodFeaturesToTrack(g_grayImage,//输入图像
		corners,//检测到的角点的输出向量
		g_maxCornerNumber,//角点的最大数量
		qualityLevel,//角点检测可接受的最小特征值
		minDistance,//角点之间的最小距离
		Mat(),//感兴趣区域
		blockSize,//计算导数自相关矩阵时指定的邻域范围
		false,//不使用Harris角点检测
		k);//权重系数

	//【4】输出文字信息
	cout << "\n\t>-------------此次检测到的角点数量为：" << corners.size() << endl;

	//【5】绘制检测到的角点
	int r = 4;
	for (unsigned int i = 0; i < corners.size(); i++)
	{
		//以随机的颜色绘制出角点
		circle(copy, corners[i], r, Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255),
			g_rng.uniform(0, 255)), -1, 8, 0);
	}

	//【6】显示（更新）窗口
	imshow(WINDOW_NAME, copy);

	//【7】亚像素角点检测的参数设置
	Size winSize = Size(5, 5);
	Size zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);

	//【8】计算出亚像素角点位置
	cornerSubPix(g_grayImage, corners, winSize, zeroZone, criteria);

	//【9】输出角点信息
	for (int i = 0; i < corners.size(); i++)
	{
		cout << " \t>>精确角点坐标[" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << endl;
		points = corners;
	}
	waitKey(0);
	destroyWindow(WINDOW_NAME);
 }
//使用上面的函数步骤坐标得到的面积不正确，每次运行得到的面积都不对！
double ContourArea(Point2f q1, Point2f q2, Point2f q3, Point2f q4) {
	vector<Point> contour;
	contour.push_back(q1);
	contour.push_back(q2);
	contour.push_back(q3);
	contour.push_back(q4);

	double area0 = contourArea(contour);
	vector<Point> approx;
	approxPolyDP(contour, approx, 5, true);
	double area1 = contourArea(approx);
	cout << area0 << endl << "area1 = " << area1 << endl << "近似多边形" << approx.size() << endl;
	return area0;
}

//-----------------------------------------------------------------------------------------------
int Threshold(Mat g_srcImage)
{
	if (!g_srcImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false; }
	imshow("原始图", g_srcImage);

	//【2】存留一份原图的灰度图

	//【3】创建窗口并显示原始图
	namedWindow("【程序窗口】", WINDOW_AUTOSIZE);

	//【4】创建滑动条来控制阈值
	createTrackbar("模式",
		"【程序窗口】", &g_nThresholdType,
		4, on_Threshold);

	createTrackbar("参数值",
		"【程序窗口】", &g_nThresholdValue,
		255, on_Threshold);

	//【5】初始化自定义的阈值回调函数
	on_Threshold(0, 0);

	// 【6】轮询等待用户按键，如果ESC键按下则退出程序
	while (1)
	{
		int key;
		key = waitKey(20);
		if ((char)key == 27) { break; }
	}

}
//-----------------------------------【on_Threshold( )函数】------------------------------------
//		描述：自定义的阈值回调函数
//-----------------------------------------------------------------------------------------------
void on_Threshold(int, void*)
{
	//调用阈值函数
	threshold(transformed, g_dstImage, g_nThresholdValue, 255, g_nThresholdType);

	//更新效果图
	imshow(WINDOW_NAME, g_dstImage);
}

//-----------------------------------------------------------------------------------------------
int OpenOperation(Mat image,MorphTypes a)
{
	Mat src = image;
	//创建窗口   
	namedWindow("【原始图】");
	namedWindow("【效果图】");
	//显示原始图  
	imshow("【原始图】", src);
	//定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	//进行形态学操作
	morphologyEx(src, image, a, element);
	//显示效果图  
	imshow("【效果图】", image);

	waitKey(0);
	destroyAllWindows();

	return 0;
}
//K—Means算法进行分割
int K_Means(Mat pic)
{
	const int MAX_CLUSTERS = 5;
	Vec3b colorTab[] =
	{
		Vec3b(255,255, 255),
		Vec3b(0, 0, 0),
		/*Vec3b(255, 100, 100),
		Vec3b(255, 0, 255),
		Vec3b(0, 255, 255)*/
	};
	Mat data, labels;
	for (int i = 0; i < pic.rows; i++)
		for (int j = 0; j < pic.cols; j++)
		{
			Vec3b point = pic.at<Vec3b>(i, j);
			Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
			data.push_back(tmp);
		}

	//根据浏览图片，确定k=2
	kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
		3, KMEANS_RANDOM_CENTERS);

	int n = 0;
	//显示聚类结果，不同的类别用不同的颜色显示
	for (int i = 0; i < pic.rows; i++)
		for (int j = 0; j < pic.cols; j++)
		{
			int clusterIdx = labels.at<int>(n);
			pic.at<Vec3b>(i, j) = colorTab[clusterIdx];
			n++;
		}
	imshow("pic", pic);
	waitKey(0);

	return 0;
}
double Contours(Mat src)
{
	Mat dst;
	if (src.empty())
	{
		printf("can not load image \n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
	dst = Mat::zeros(src.size(), CV_8UC3);
	//前面经过处理无需再次进行
	//blur(src, src, Size(3, 3));
	//Canny(src, src, 20, 80, 3, false);
	cvtColor(src, src, COLOR_BGR2GRAY);
	
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	RNG rng(0);
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//	drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point(0, 0));
	//}//因为只需要最内部的轮廓所以不需要显示所以的轮廓

	double g_dConArea = contourArea(contours[0]);
	cout << "树叶的面积为：" << g_dConArea << endl;

	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(dst, contours, 0, color, 2, 8, hierarchy, 0, Point(0, 0));//参数0表示显示树叶轮廓
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", dst);
	waitKey();
	destroyWindow("output");
	destroyWindow("input");
	return g_dConArea;
}
double RectArea(Mat srcImage,Point2f q1, Point2f q2, Point2f q3, Point2f q4) {
	//一下是对点的值进行测试的代码
	Mat src;
	srcImage.copyTo(src);
	cout << "检验q的排序是否正确" <<q1 << q2 << q3 << q4 << endl;
	//const Point* ppt[1] = { root_points[0] };
	//int npt[] = { 4 };
	//polylines(src, ppt, npt, 1, 1, Scalar(255, 255, 255), 1, 8, 0);
	rectangle(src, q1, q4, Scalar(0, 0, 255), -1, 8);
	imshow("dstImage", src);
	imshow("srcImage", srcImage);
	waitKey();
	destroyWindow("Test");
	destroyWindow("yuan");


	Mat dst;
	if (src.empty())
	{
		printf("can not load image \n");
		return -1;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", src);
	dst = Mat::zeros(src.size(), CV_8UC3);

	blur(src, src, Size(3, 3));
	cvtColor(src, src, COLOR_BGR2GRAY);
	Canny(src, src, 20, 80, 3, false);
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	RNG rng(0);
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//	drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point(0, 0));
	//}//因为只需要最内部的轮廓所以不需要显示所以的轮廓
	//计算轮廓的面积 ?
	cout << "【筛选前总共轮廓个数为】：" << (int)contours.size() << endl;
	//以下代码用来验证是否计算的时矩形的面积
	////计算轮廓的面积 ?
	//cout << "【筛选前总共轮廓个数为】：" << (int)contours.size() << endl;
	//for (int i = 0; i < (int)contours.size(); i++)
	//{
	//	double g_dConArea = contourArea(contours[i], true);
	//	cout << "【用轮廓面积计算函数计算出来的第" << i << "个轮廓的面积为：】" << g_dConArea << endl;
	//}
	double g_dConArea = contourArea(contours[0]);
	cout << "矩形的面积：" << g_dConArea << endl;

	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(dst, contours, 0, color, 2, 8, hierarchy, 0, Point(0, 0));//参数0表示显示树叶轮廓
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", dst);
	waitKey();
	//vector<Point> contour;
	//double area = contourArea(contour);
	//cout << "矩形的面积为：" << area << endl;

	return g_dConArea;
}

//subfunction
bool cmpy(cv::Point const& a, cv::Point const& b)
{
	return a.y < b.y;
}
bool cmpx(cv::Point const& a, cv::Point const& b)
{
	Point point;
	if (a.x < b.x) {
		return 0;
	}
	else {
		point = points[0];
		points[0] = points[1];
		points[1] = point;
	}
}


int RectMask(Mat image,double e)
{
	Mat src;
	image.copyTo(src);
	Point root_points[1][4];
	root_points[0][0] = Point(e, e);
	root_points[0][1] = Point(src.rows-e, e);
	root_points[0][2] = Point(src.rows - e, src.cols - e);
	root_points[0][3] = Point(e, src.cols-e);

	const Point* ppt[1] = { root_points[0] };
	int npt[] = { 4 };

	cv::Mat mask_ann, dst;
	src.copyTo(mask_ann);
	mask_ann.setTo(cv::Scalar::all(0));

	fillPoly(mask_ann, ppt, npt, 1, Scalar(255, 255, 255));

	src.copyTo(dst, mask_ann);
	imshow("dst", dst);
	dst.copyTo(image);
	//imshow("src", image);
	//imshow("mask_ann", mask_ann);//用于测试mask是否运行良好
	waitKey(0);
	destroyAllWindows();
	return 0;
}

