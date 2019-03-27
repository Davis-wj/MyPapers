#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

#define WINDOW_NAME "�����򴰿ڡ�" 
#define Patch "E:\\TestDate\\hudeilan\\Date\\У�����ͼ��.jpg"
//E:\\TestDate\\verify\\01.jpg   ������֤��ȷ��
//-----------------------------------��ȫ�ֱ����������֡�-------------------------------------------
//		������ȫ�ֱ���������
Point g_Point;
vector<Point2f> points(4);	//���������꣬���ڿ����Ƿ�����Ż�
bool g_bPoint = false;//�Ƿ���л���
RNG g_rng(12345);
int cnt = 0;//��ʼ�����ô���
Mat transformed;	//�洢͸�ӱ任������ͼƬ
double RectA4 = 100.0;
//-----------------------------------------------------------------------------------------------
int g_nThresholdValue = 100;
int g_nThresholdType = 3;
Mat g_dstImage;
double e = 35;//����ROI���붥��ı߽�Ϊ����
//-----------------------------------------------------------------------------------------------

void GetCoordinates(Mat srcImage);
void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawFilledCircle(cv::Mat& img, Point center);

void tf(Point2f q1, Point2f q2, Point2f q3, Point2f q4);
void HSVSplit(Mat image);
void GoodFeaturesToTrack(Mat g_srcImage, double minDistance);
double ContourArea(Point2f q1, Point2f q2, Point2f q3, Point2f q4);
void on_Threshold(int, void*);//�ص�����
int Threshold(Mat g_srcImage);//��ֵ��
int OpenOperation(Mat image, MorphTypes a);
int K_Means(Mat pic);//K-Means�㷨���зָ�
double Contours(Mat src);
double RectArea(Mat src,Point2f q1, Point2f q2, Point2f q3, Point2f q4);
bool cmpy(cv::Point const& a, cv::Point const& b);//�Ե��yxֵ���д�С���������
bool cmpx(cv::Point const& a, cv::Point const& b);
int RectMask(Mat image, double e);
int main() {
	transformed = imread(Patch);
	GetCoordinates(transformed);
 	waitKey(0);
	destroyWindow(WINDOW_NAME);
	//͸�ӱ任
	tf(points[0], points[1], points[2], points[3]);
	waitKey(0);
	//ת��HSVɫ�ȿռ���зָ�
	HSVSplit(transformed);//K-Means�㷨����
	//K_Means(transformed);


	//ȷ���ĵ������
	double minDistance = 350;//�ǵ�֮�����С����
	GoodFeaturesToTrack(transformed, minDistance);
	//OpenOperation(transformed, MORPH_OPEN);//����Ҫ��ֵ����
	//OpenOperation(transformed, MORPH_CLOSE);//��������

	sort(points.begin(), points.end(), cmpy);//�Ե��yֵ���д�С���������,��ȷ�����ϽǺ����½ǵ�����
	cmpx(points[0],points[1]);//������yֵ��ǰ��������xֵ������
	cmpx(points[2], points[3]);//������yֵ�ĺ���������xֵ������
	//������ε����
	double R = RectArea(transformed, points[0], points[1], points[2], points[3]);
	//������Ҷ�����
	double L = Contours(transformed);
	double area = RectA4 * L / R;
	cout << "��Ҷ����ʵ���Ϊ" << area << endl;
	//Contours(transformed);
	//��ֵ��
	//Threshold(transformed);
	//GoodFeaturesToTrack(transformed);
	waitKey(0);
	
	//������ε����


}

void PerspectiveTransform() {
	cout << "�����Զ�͸�ӱ任" << endl;

}
//-----------------------------------------------------------------------------------------------
//GetCoordinates();ʹ�õĺ���
void GetCoordinates(Mat srcImage) {
	Mat tempImage;

	srcImage.copyTo(tempImage);

	//��2�������������ص�����
	namedWindow(WINDOW_NAME);
	setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);

	while (1)
	{
		srcImage.copyTo(tempImage);//����Դͼ����ʱ����
		if (g_bPoint) DrawFilledCircle(tempImage, g_Point);//�����л��Ƶı�ʶ��Ϊ�棬����л���
		imshow(WINDOW_NAME, tempImage);
		if (waitKey(10) == 27 || cnt > 3) break;//����ESC�������ߵ��Ĵ����������˳�
	}
}
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{

	Mat& image = *(cv::Mat*) param;
	switch (event)
	{
		//���������Ϣ
	case EVENT_LBUTTONDOWN:
	{
		g_bPoint = true;
		g_Point = Point2f(x, y);//��¼��ʼ��
	}
	break;

	//���̧����Ϣ
	case EVENT_LBUTTONUP:
	{
		g_bPoint = false;//�ñ�ʶ��Ϊfalse
		//�Կ�͸�С��0�Ĵ���
		++cnt;//��ȡ�������õĴ���
		DrawFilledCircle(image, g_Point);
		cout << "��" << cnt << "���� " << "����Ϊ��" << g_Point << endl;
		int i = cnt - 1;
		points[i] = g_Point;

		//����ע�ʹ�����������points�Ƿ�����
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
//����͸�ӱ任
void tf(Point2f q1, Point2f q2, Point2f q3, Point2f q4) {
	Mat src = imread(Patch);
	Point Q1 = q1;
	Point Q2 = q2;
	Point Q3 = q3;
	Point Q4 = q4;

	//���Դ���
	//Point Q1 = Point2f(322, 242);
	//Point Q2 = Point2f(638, 348);
	//Point Q3 = Point2f(539, 530);
	//Point Q4 = Point2f(188, 375);

	// compute the size of the card by keeping aspect ratio.
	double ratio = 1.0;//����ϵ����Ϊ�ĳ���ģ��Ϊ ������ ����ֵ����Ϊ1.0
	double cardH = sqrt((Q3.x - Q2.x)*(Q3.x - Q2.x) + (Q3.y - Q2.y)*(Q3.y - Q2.y));//��������Ը����Լ������
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
	warpPerspective(src, transformed, transmtx, src.size(),1,1);//ԭ��transmtx.size()

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
	cvtColor(image, hsvimage,COLOR_BGR2HSV); //RGB��HSV��ɫ�ռ��ת��
	imshow("HSV", hsvimage); //ֱ�Ӱ�HSV����ͼ����RGB��ʽ��ʾ����ʾ������ͼ�����ԭͼ��ͬ
	vector<Mat> hsv;
	split(hsvimage, hsv);//��HSV����ͨ������
	imshow("Sɫ�ȿռ�", hsv.at(1));
	transformed = hsv.at(1);
	waitKey();
	destroyWindow("image");
	destroyWindow("HSV");
	destroyWindow("Sɫ�ȿռ�");
}
void GoodFeaturesToTrack(Mat g_srcImage,double minDistance) {
	Mat g_grayImage = g_srcImage.clone();
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
	int g_maxCornerNumber = 4;
	//��1���Ա���С�ڵ���1ʱ�Ĵ���
	if (g_maxCornerNumber <= 1) { g_maxCornerNumber = 1; }

	//��2��Shi-Tomasi�㷨��goodFeaturesToTrack�������Ĳ���׼��
	vector<Point2f> corners;
	double qualityLevel = 0.01;//�ǵ���ɽ��ܵ���С����ֵ

	int blockSize = 3;//���㵼������ؾ���ʱָ��������Χ
	double k = 0.04;//Ȩ��ϵ��
	Mat copy = g_srcImage.clone();	//����Դͼ��һ����ʱ�����У���Ϊ����Ȥ����

	//д��ROI����	
	RectMask(g_grayImage, e);
	//��3������Shi-Tomasi�ǵ���
	goodFeaturesToTrack(g_grayImage,//����ͼ��
		corners,//��⵽�Ľǵ���������
		g_maxCornerNumber,//�ǵ���������
		qualityLevel,//�ǵ���ɽ��ܵ���С����ֵ
		minDistance,//�ǵ�֮�����С����
		Mat(),//����Ȥ����
		blockSize,//���㵼������ؾ���ʱָ��������Χ
		false,//��ʹ��Harris�ǵ���
		k);//Ȩ��ϵ��

	//��4�����������Ϣ
	cout << "\n\t>-------------�˴μ�⵽�Ľǵ�����Ϊ��" << corners.size() << endl;

	//��5�����Ƽ�⵽�Ľǵ�
	int r = 4;
	for (unsigned int i = 0; i < corners.size(); i++)
	{
		//���������ɫ���Ƴ��ǵ�
		circle(copy, corners[i], r, Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255),
			g_rng.uniform(0, 255)), -1, 8, 0);
	}

	//��6����ʾ�����£�����
	imshow(WINDOW_NAME, copy);

	//��7�������ؽǵ���Ĳ�������
	Size winSize = Size(5, 5);
	Size zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);

	//��8������������ؽǵ�λ��
	cornerSubPix(g_grayImage, corners, winSize, zeroZone, criteria);

	//��9������ǵ���Ϣ
	for (int i = 0; i < corners.size(); i++)
	{
		cout << " \t>>��ȷ�ǵ�����[" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << endl;
		points = corners;
	}
	waitKey(0);
	destroyWindow(WINDOW_NAME);
 }
//ʹ������ĺ�����������õ����������ȷ��ÿ�����еõ�����������ԣ�
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
	cout << area0 << endl << "area1 = " << area1 << endl << "���ƶ����" << approx.size() << endl;
	return area0;
}

//-----------------------------------------------------------------------------------------------
int Threshold(Mat g_srcImage)
{
	if (!g_srcImage.data) { printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ����ͼƬ����~�� \n"); return false; }
	imshow("ԭʼͼ", g_srcImage);

	//��2������һ��ԭͼ�ĻҶ�ͼ

	//��3���������ڲ���ʾԭʼͼ
	namedWindow("�����򴰿ڡ�", WINDOW_AUTOSIZE);

	//��4��������������������ֵ
	createTrackbar("ģʽ",
		"�����򴰿ڡ�", &g_nThresholdType,
		4, on_Threshold);

	createTrackbar("����ֵ",
		"�����򴰿ڡ�", &g_nThresholdValue,
		255, on_Threshold);

	//��5����ʼ���Զ������ֵ�ص�����
	on_Threshold(0, 0);

	// ��6����ѯ�ȴ��û����������ESC���������˳�����
	while (1)
	{
		int key;
		key = waitKey(20);
		if ((char)key == 27) { break; }
	}

}
//-----------------------------------��on_Threshold( )������------------------------------------
//		�������Զ������ֵ�ص�����
//-----------------------------------------------------------------------------------------------
void on_Threshold(int, void*)
{
	//������ֵ����
	threshold(transformed, g_dstImage, g_nThresholdValue, 255, g_nThresholdType);

	//����Ч��ͼ
	imshow(WINDOW_NAME, g_dstImage);
}

//-----------------------------------------------------------------------------------------------
int OpenOperation(Mat image,MorphTypes a)
{
	Mat src = image;
	//��������   
	namedWindow("��ԭʼͼ��");
	namedWindow("��Ч��ͼ��");
	//��ʾԭʼͼ  
	imshow("��ԭʼͼ��", src);
	//�����
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	//������̬ѧ����
	morphologyEx(src, image, a, element);
	//��ʾЧ��ͼ  
	imshow("��Ч��ͼ��", image);

	waitKey(0);
	destroyAllWindows();

	return 0;
}
//K��Means�㷨���зָ�
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

	//�������ͼƬ��ȷ��k=2
	kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
		3, KMEANS_RANDOM_CENTERS);

	int n = 0;
	//��ʾ����������ͬ������ò�ͬ����ɫ��ʾ
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
	//ǰ�澭�����������ٴν���
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
	//}//��Ϊֻ��Ҫ���ڲ����������Բ���Ҫ��ʾ���Ե�����

	double g_dConArea = contourArea(contours[0]);
	cout << "��Ҷ�����Ϊ��" << g_dConArea << endl;

	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(dst, contours, 0, color, 2, 8, hierarchy, 0, Point(0, 0));//����0��ʾ��ʾ��Ҷ����
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", dst);
	waitKey();
	destroyWindow("output");
	destroyWindow("input");
	return g_dConArea;
}
double RectArea(Mat srcImage,Point2f q1, Point2f q2, Point2f q3, Point2f q4) {
	//һ���ǶԵ��ֵ���в��ԵĴ���
	Mat src;
	srcImage.copyTo(src);
	cout << "����q�������Ƿ���ȷ" <<q1 << q2 << q3 << q4 << endl;
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
	//}//��Ϊֻ��Ҫ���ڲ����������Բ���Ҫ��ʾ���Ե�����
	//������������� ?
	cout << "��ɸѡǰ�ܹ���������Ϊ����" << (int)contours.size() << endl;
	//���´���������֤�Ƿ�����ʱ���ε����
	////������������� ?
	//cout << "��ɸѡǰ�ܹ���������Ϊ����" << (int)contours.size() << endl;
	//for (int i = 0; i < (int)contours.size(); i++)
	//{
	//	double g_dConArea = contourArea(contours[i], true);
	//	cout << "��������������㺯����������ĵ�" << i << "�����������Ϊ����" << g_dConArea << endl;
	//}
	double g_dConArea = contourArea(contours[0]);
	cout << "���ε������" << g_dConArea << endl;

	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(dst, contours, 0, color, 2, 8, hierarchy, 0, Point(0, 0));//����0��ʾ��ʾ��Ҷ����
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", dst);
	waitKey();
	//vector<Point> contour;
	//double area = contourArea(contour);
	//cout << "���ε����Ϊ��" << area << endl;

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
	//imshow("mask_ann", mask_ann);//���ڲ���mask�Ƿ���������
	waitKey(0);
	destroyAllWindows();
	return 0;
}

