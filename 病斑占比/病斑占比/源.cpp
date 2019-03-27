#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include<fstream>

//-----------------------------------【命名空间声明部分】---------------------------------------  
//      描述：包含程序所使用的命名空间  
//-----------------------------------------------------------------------------------------------   
using namespace cv;
using namespace std;
#define SavePatch "D:\\病斑占比\\测试\\测试02.txt"
#define Patch "D:\\病斑占比\\数据\\P (2).JPG"
string SaveImageDirPatch = "D:\\病斑占比\\测试\\";
string SaveImageName = "测试02 ";
#define WINDOW_NAME "【程序窗口】" 

//-----------------------------------【全局变量声明部分】--------------------------------------  
//      描述：全局变量声明  
//-----------------------------------------------------------------------------------------------  
double RectA4 = 39601;	//该值用于记录矩形的真实面积值
double reactArea = 1800; //假定矩形像素值
Mat g_srcImage, g_dstImage, g_grayImage, g_maskImage, dstmask;//定义原始图、目标图、灰度图、掩模图
int g_nFillMode = 1;//漫水填充的模式
int g_nLowDifference = 47, g_nUpDifference = 46;//负差最大值、正差最大值
int g_nConnectivity = 4;//表示floodFill函数标识符低八位的连通值
int g_bIsColor = true;//是否为彩色图的标识符布尔值
bool g_bUseMask = false;//是否显示掩膜窗口的布尔值
int g_nNewMaskVal = 255;//新的重新绘制的像素值
int area;	//漫水填充 选定的像素值
vector<Point2f> points(4);	//储存点的坐标，后期看看是否可以优化
Point g_Point;	//用于on_MouseHandle()函数进行坐标点的存储
bool g_bPoint = false;//on_MouseHandle()函数是否进行绘制
int cnt = 0;	//初始化调用次数在on_MouseHandle()等函数进行使用

//-----------------------------------【函数的声明】----------------------------------  
//      描述：输出一些帮助信息  
//---------------------------------------------------------------------------------------------- 
void GetCoordinates(Mat srcImage);
void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawFilledCircle(cv::Mat& img, Point center);

void tf(Point2f q1, Point2f q2, Point2f q3, Point2f q4);
void HSVSplit(Mat image);//V1.0进行了更改
int K_Means(Mat pic);//K-Means算法进行分割
double Contours(Mat src);
void saveImage(string savePatch, Mat image);
//int *ptr =  &area;

//-----------------------------------【ShowHelpText( )函数】----------------------------------  
//      描述：输出一些帮助信息  
//----------------------------------------------------------------------------------------------  
static void ShowHelpText()
{
	//输出一些帮助信息  
	printf("\n\n\n\t欢迎来到漫水填充示例程序~\n\n");
	printf("\n\n\t按键操作说明: \n\n"
		"\t\t鼠标点击图中区域- 进行漫水填充操作\n"
		"\t\t键盘按键【ESC】- 退出程序\n"
		"\t\t键盘按键【1】-  切换彩色图/灰度图模式\n"
		"\t\t键盘按键【2】- 显示/隐藏掩膜窗口\n"
		"\t\t键盘按键【3】- 恢复原始图像\n"
		"\t\t键盘按键【4】- 使用空范围的漫水填充\n"
		"\t\t键盘按键【5】- 使用渐变、固定范围的漫水填充\n"
		"\t\t键盘按键【6】- 使用渐变、浮动范围的漫水填充\n"
		"\t\t键盘按键【7】- 操作标志符的低八位使用4位的连接模式\n"
		"\t\t键盘按键【8】- 操作标志符的低八位使用8位的连接模式\n"
	);
}


//-----------------------------------【onMouse( )函数】--------------------------------------  
//      描述：鼠标消息onMouse回调函数
//---------------------------------------------------------------------------------------------
static void onMouse(int event, int x, int y, int, void*)
{
	// 若鼠标左键没有按下，便返回
	if (event != EVENT_LBUTTONDOWN)
		return;

	//-------------------【<1>调用floodFill函数之前的参数准备部分】---------------
	Point seed = Point(x, y);
	int LowDifference = g_nFillMode == 0 ? 0 : g_nLowDifference;//空范围的漫水填充，此值设为0，否则设为全局的g_nLowDifference
	int UpDifference = g_nFillMode == 0 ? 0 : g_nUpDifference;//空范围的漫水填充，此值设为0，否则设为全局的g_nUpDifference
	int flags = g_nConnectivity + (g_nNewMaskVal << 8) +
		(g_nFillMode == 1 ? FLOODFILL_FIXED_RANGE : 0);//标识符的0~7位为g_nConnectivity，8~15位为g_nNewMaskVal左移8位的值，16~23位为CV_FLOODFILL_FIXED_RANGE或者0。

	//随机生成bgr值
	int b = 255;//随机返回一个0~255之间的值
	int g = 255;//随机返回一个0~255之间的值
	int r = 255;//随机返回一个0~255之间的值
	Rect ccomp;//定义重绘区域的最小边界矩形区域

	Scalar newVal = g_bIsColor ? Scalar(b, g, r) : Scalar(r*0.299 + g * 0.587 + b * 0.114);//在重绘区域像素的新值，若是彩色图模式，取Scalar(b, g, r)；若是灰度图模式，取Scalar(r*0.299 + g*0.587 + b*0.114)

	Mat dst = g_bIsColor ? g_dstImage : g_grayImage;//目标图的赋值


	//--------------------【<2>正式调用floodFill函数】-----------------------------
	if (g_bUseMask)
	{
		threshold(g_maskImage, g_maskImage, 0, 255, THRESH_BINARY);
		area = floodFill(dst, g_maskImage, seed, newVal, &ccomp, Scalar(LowDifference, LowDifference, LowDifference),
			Scalar(UpDifference, UpDifference, UpDifference), flags);
		imshow("mask", g_maskImage);
	}
	else
	{
		area = floodFill(dst, seed, newVal, &ccomp, Scalar(LowDifference, LowDifference, LowDifference),
			Scalar(UpDifference, UpDifference, UpDifference), flags);
	}

	imshow("效果图", dst);
	dstmask.copyTo(g_maskImage);
	saveImage(SaveImageDirPatch + SaveImageName + "_效果图.jpg", dst);
	//saveImage(SaveImageDirPatch + SaveImageName + "_mask.jpg", dstmask);
	//ptr = &area;
	cout << area << " 个像素被重绘\n";
}

//-----------------------------------【Contours( )函数】--------------------------------------  
//      描述：识别树叶的轮廓，函数内计算了树叶轮廓的面积并进行值的返回
//---------------------------------------------------------------------------------------------
double Contours(Mat src)
{
	Mat dst;
	if (src.empty())
	{
		printf("can not load image \n");
		return -1;
	}
	dst = Mat::zeros(src.size(), CV_8UC3);

	cvtColor(src, src, COLOR_BGR2GRAY);
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));//RETR_EXTERNAL:只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
																						//CHAIN_APPROX_SIMPLE 仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours 向量内，拐点与拐点之间直线段上的信息点不予保留
																						//CHAIN_APPROX_SIMPLE 仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours 向量内，拐点与拐点之间直线段上的信息点不予保留

	RNG rng(0);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(dst, contours, i, color, 2, LINE_AA, hierarchy, 0, Point(0, 0));
		//drawContours(
		//InputOutputArray  binImg, // 输出图像
		//OutputArrayOfArrays  contours,//  全部发现的轮廓对象
		//Int contourIdx// 轮廓索引号
		//const Scalar & color,// 绘制时候颜色
		//int  thickness,// 绘制线宽
		//int  lineType,// 线的类型LINE_8
		//InputArray hierarchy,// 拓扑结构图
		//int maxlevel,// 最大层数， 0只绘制当前的，1表示绘制绘制当前及其内嵌的轮廓
		//Point offset = Point()// 轮廓位移，可选
	}

	imshow("树叶的轮廓图", dst);
	saveImage(SaveImageDirPatch + SaveImageName + "_轮廓图.jpg", dst);
	waitKey(0);
	double ConArea = 0.0;
	for (int i = 1; i < contours.size(); i++)
	{
		ConArea += contourArea(contours[i]);
	}
	return ConArea;
}

//-----------------------------------【HSVSplit( )函数】--------------------------------------  
//      描述：分割图像色度空间
//---------------------------------------------------------------------------------------------
void HSVSplit(Mat image) {
	Mat hsvimage, hue;
	imshow("image", image);
	cvtColor(image, hsvimage, COLOR_BGR2HSV); //RGB到HSV颜色空间的转换
	//imshow("HSV", hsvimage); //直接把HSV规格的图像以RGB格式显示。显示出来的图像会与原图不同
	vector<Mat> hsv;
	split(hsvimage, hsv);//将HSV三个通道分离
	imshow("S色度空间", hsv.at(1));
	saveImage(SaveImageDirPatch + SaveImageName + "_S色度空间.jpg", hsv.at(1));

	g_srcImage = hsv.at(1);
	hsv[0] = 0;
	hsv[2] = 0;
	merge(hsv, g_srcImage);

	cvtColor(g_srcImage, g_srcImage, COLOR_BGR2RGB);
	waitKey();
	//destroyWindow("image");
	//destroyWindow("HSV");
	//destroyWindow("S色度空间");

}

//-----------------------------------【K_Means( )函数】--------------------------------------  
//      描述：使用K_Means进行图像的聚类，这里设置的是2类
//---------------------------------------------------------------------------------------------
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
	saveImage(SaveImageDirPatch + SaveImageName + "_K.jpg", pic);
	waitKey(0);

	return 0;
}

//-----------------------------------【tf( )函数】--------------------------------------  
//      描述：使用透视变换对图像进行校正
//---------------------------------------------------------------------------------------------
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
	/*cout <<"Test"<< squre_pts[0] << squre_pts[1] << squre_pts[2] << squre_pts[3] << endl;*/

	//将points的坐标更新为转换后的坐标
	for (int i = 0; i < 4; i++) {
		points[i] = squre_pts[i];
		cout << "检测point:" << "points[" << i << "]:" << points[i] << endl;
	}
	int offsetSize = 150;
	g_srcImage = Mat::zeros(R.height + offsetSize, R.width + offsetSize, CV_8UC3);
	warpPerspective(src, g_srcImage, transmtx, src.size(), 1, 1);//原是transmtx.size()

	//rectangle(src, R, Scalar(0,255,0),1,8,0);

	line(src, Q1, Q2, Scalar(0, 0, 255), 1, LINE_AA, 0);
	line(src, Q2, Q3, Scalar(0, 0, 255), 1, LINE_AA, 0);
	line(src, Q3, Q4, Scalar(0, 0, 255), 1, LINE_AA, 0);
	line(src, Q4, Q1, Scalar(0, 0, 255), 1, LINE_AA, 0);

	imshow("quadrilateral", g_srcImage);
	imshow("src", src);
	imwrite("tf1_1.jpg", g_srcImage);

	waitKey();
	destroyWindow("quadrilateral");
	destroyWindow("src");
}

//-----------------------------------【GetCoordinates( )、DrawFilledCircle（）与on_MouseHandle（）函数】--------------------------------------  
//      描述：获取特征点的坐标用于上面tf（）函数的透视变换，DrawFilledCircle（）函数用于画点，定义了点的大小等数据
//---------------------------------------------------------------------------------------------
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
	circle(img, center, 10, Scalar(100, 0, 100), thickness, lineType);

}

//-----------------------------------【saveImage（）函数】--------------------------------------  
//      描述：将运行的图片进行保存
//---------------------------------------------------------------------------------------------
void saveImage(string savePatch, Mat image) {
	//name 写入保存的路径
	//image 出入的图像
	imwrite(savePatch, image);
}

//-----------------------------------【下面函数】--------------------------------------  
//      描述：将删除面积小的区域
//---------------------------------------------------------------------------------------------

//轮廓按照面积大小升序排序
bool ascendSort(vector<Point> a, vector<Point> b) {
	return a.size() < b.size();

}

//轮廓按照面积大小降序排序
bool descendSort(vector<Point> a, vector<Point> b) {
	return a.size() > b.size();
}

//除去小轮廓的部分
void ContourRemoval(Mat image, int a) {
	Mat srcImage = image;
	Mat thresholdImage;
	Mat grayImage;
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, thresholdImage, 0, 255, THRESH_OTSU + THRESH_BINARY);
	//Mat resultImage;
	//thresholdImage.copyTo(resultImage);
	vector< vector< Point> > contours;  //用于保存所有轮廓信息
	vector< vector< Point> > contours2; //用于保存面积不足100的轮廓
	vector<Point> tempV;				//暂存的轮廓

	findContours(thresholdImage, contours, RETR_LIST, CHAIN_APPROX_NONE);
	//cv::Mat labels;
	//int N = connectedComponents(resultImage, labels, 8, CV_16U);
	//findContours(labels, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//轮廓按照面积大小进行升序排序
	sort(contours.begin(), contours.end(), ascendSort);//升序排序
	vector<vector<Point> >::iterator itc = contours.begin();
	int i = 0;
	while (itc != contours.end())
	{
		//获得轮廓的矩形边界
		Rect rect = boundingRect(*itc);
		int x = rect.x;
		int y = rect.y;
		int w = rect.width;
		int h = rect.height;
		//绘制轮廓的矩形边界
		cv::rectangle(srcImage, rect, { 0, 0, 255 }, 1);
		//保存图片
		char str[10];
		//printf(str, "%d.jpg", i++);
		cv::imshow("srcImage", srcImage);
		//imwrite(str, srcImage);
		waitKey(10);//将参数设置为1000可以看见除去小块轮廓的过程

		if (itc->size() < a)
		{
			//把轮廓面积不足100的区域，放到容器contours2中，
			tempV.push_back(Point(x, y));
			tempV.push_back(Point(x, y + h));
			tempV.push_back(Point(x + w, y + h));
			tempV.push_back(Point(x + w, y));
			contours2.push_back(tempV);
			/*也可以直接用：contours2.push_back(*itc);代替上面的5条语句*/
			//contours2.push_back(*itc);

			//删除轮廓面积不足100的区域，即用黑色填充轮廓面积不足100的区域：
			cv::drawContours(srcImage, contours2, -1, Scalar(255, 255, 255), FILLED);
		}
		//保存图片
		//printf(str, "%d.jpg", i++);
		cv::imshow("srcImage", srcImage);
		//imwrite(str, srcImage);
		cv::waitKey(1);
		tempV.clear();
		++itc;
	}
	waitKey(0);
}

//-----------------------------------【main( )函数】--------------------------------------------  
//      描述：控制台应用程序的入口函数，我们的程序从这里开始  
//-----------------------------------------------------------------------------------------------  
int main(int argc, char** argv)
{
	ofstream out;
	out.open(SavePatch, ios::trunc);
	//改变console字体颜色  
	system("color 2F");

	//载入原图
	g_srcImage = imread(Patch, 1);

	if (!g_srcImage.data) { printf("Oh，no，读取图片image0错误~！ \n"); return false; }

	//显示帮助文字
	ShowHelpText();

	g_srcImage.copyTo(g_dstImage);//拷贝源图到目标图
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);//转换三通道的image0到灰度图
	g_maskImage.create(g_srcImage.rows + 2, g_srcImage.cols + 2, CV_8UC1);//利用image0的尺寸来初始化掩膜mask

	namedWindow("效果图", WINDOW_AUTOSIZE);

	//创建Trackbar
	createTrackbar("负差最大值", "效果图", &g_nLowDifference, 255, 0);
	createTrackbar("正差最大值", "效果图", &g_nUpDifference, 255, 0);

	//鼠标回调函数
	setMouseCallback("效果图", onMouse, 0);

	//循环轮询按键
	while (1)
	{
		//先显示效果图
		imshow("效果图", g_bIsColor ? g_dstImage : g_grayImage);

		//获取键盘按键
		int c = waitKey(0);
		//判断ESC是否按下，若按下便退出
		if ((c & 255) == 27)
		{
			cout << "程序退出...........\n";
			dstmask.copyTo(g_maskImage);
			break;
		}

		//根据按键的不同，进行各种操作
		switch ((char)c)
		{
			//如果键盘“1”被按下，效果图在在灰度图，彩色图之间互换
		case '1':
			if (g_bIsColor)//若原来为彩色，转为灰度图，并且将掩膜mask所有元素设置为0
			{
				cout << "键盘“1”被按下，切换彩色/灰度模式，当前操作为将【彩色模式】切换为【灰度模式】\n";
				cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
				g_maskImage = Scalar::all(0);	//将mask所有元素设置为0
				g_bIsColor = false;	//将标识符置为false，表示当前图像不为彩色，而是灰度
			}
			else//若原来为灰度图，便将原来的彩图image0再次拷贝给image，并且将掩膜mask所有元素设置为0
			{
				cout << "键盘“1”被按下，切换彩色/灰度模式，当前操作为将【彩色模式】切换为【灰度模式】\n";
				g_srcImage.copyTo(g_dstImage);
				g_maskImage = Scalar::all(0);
				g_bIsColor = true;//将标识符置为true，表示当前图像模式为彩色
			}
			break;
			//如果键盘按键“2”被按下，显示/隐藏掩膜窗口
		case '2':
			if (g_bUseMask)
			{
				destroyWindow("mask");
				g_bUseMask = false;
			}
			else
			{
				namedWindow("mask", 0);
				g_maskImage = Scalar::all(0);
				imshow("mask", g_maskImage);
				g_bUseMask = true;
			}
			break;
			//如果键盘按键“3”被按下，恢复原始图像
		case '3':
			cout << "按键“3”被按下，恢复原始图像\n";
			g_srcImage.copyTo(g_dstImage);
			cvtColor(g_dstImage, g_grayImage, COLOR_BGR2GRAY);
			g_maskImage = Scalar::all(0);
			break;
			//如果键盘按键“4”被按下，使用空范围的漫水填充
		case '4':
			cout << "按键“4”被按下，使用空范围的漫水填充\n";
			g_nFillMode = 0;
			break;
			//如果键盘按键“5”被按下，使用渐变、固定范围的漫水填充
		case '5':
			cout << "按键“5”被按下，使用渐变、固定范围的漫水填充\n";
			g_nFillMode = 1;
			break;
			//如果键盘按键“6”被按下，使用渐变、浮动范围的漫水填充
		case '6':
			cout << "按键“6”被按下，使用渐变、浮动范围的漫水填充\n";
			g_nFillMode = 2;
			break;
			//如果键盘按键“7”被按下，操作标志符的低八位使用4位的连接模式
		case '7':
			cout << "按键“7”被按下，操作标志符的低八位使用4位的连接模式\n";
			g_nConnectivity = 4;
			break;
			//如果键盘按键“8”被按下，操作标志符的低八位使用8位的连接模式
		case '8':
			cout << "按键“8”被按下，操作标志符的低八位使用8位的连接模式\n";
			g_nConnectivity = 8;
			break;
		}
	}

	////获取特征点的坐标用于上面tf（）函数的透视变换
	//GetCoordinates(g_srcImage);
	//waitKey(0);
	//destroyWindow(WINDOW_NAME);
	////对图像进行透视变换
	//tf(points[0], points[1], points[2], points[3]);
	////计算矩形的面积值，并画出矩形。
	//double reactX = abs(points[1].x - points[0].x);
	//double reactY = abs(points[3].y - points[1].y);
	//double reactArea = reactX * reactY;
	//cout << "矩形占用像素面积为:" << reactArea << endl;
	//out << "矩形占用像素面积为:" << reactArea << endl;

	Mat dstRect = g_srcImage.clone();
	rectangle(dstRect, points[0], points[2], (0, 0, 255), 5);	//画出矩形
	imshow("识别的矩形", dstRect);
	saveImage(SaveImageDirPatch + SaveImageName + "_识别的矩形.jpg", dstRect);

	waitKey(0);
	destroyWindow("识别的矩形");
	//使用HSV进行图像分割
	HSVSplit(g_srcImage);
	//K-Means进行对图像的二值化和分类
	K_Means(g_srcImage);
	//腐蚀操作
	Mat element = getStructuringElement(MORPH_RECT, Size(2, 5));//定义核
	erode(g_srcImage, g_srcImage, element);
	imshow("二值化", g_srcImage);
	//除去小轮廓部分，程序设置值为300
	ContourRemoval(g_srcImage, 50);
	saveImage(SaveImageDirPatch + SaveImageName + "_二值化.jpg", g_srcImage);

	waitKey(0);
	destroyAllWindows();
	//识别树叶的轮廓，函数内计算了树叶轮廓的面积并进行值的返回	
	double ConArea = Contours(g_srcImage);
	cout << "树叶的像素为：" << ConArea << "\n" << "树叶病害像素为：" << (ConArea - area) << "\n" << "树叶的病害的比例为：" << (ConArea - area) / ConArea << endl;
	out << "树叶的像素为：" << ConArea << "\n" << "树叶病害像素为：" << (ConArea - area) << "\n" << "树叶的病害的比例为：" << (ConArea - area) / ConArea << endl;
	//计算树叶的真实面积
	double lArea = RectA4 * ConArea / reactArea;
	double pArea = RectA4 * (ConArea - area) / reactArea;
	cout << "树叶的真实面积为：" << lArea << "\n" << "树叶病害真实面积为：" << pArea << endl;
	out << "树叶的真实面积为：" << lArea << "\n" << "树叶病害真实面积为：" << pArea << endl;
	waitKey(0);


	out.close();

	return 0;
}
