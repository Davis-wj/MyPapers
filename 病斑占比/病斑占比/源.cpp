#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include<fstream>

//-----------------------------------�������ռ��������֡�---------------------------------------  
//      ����������������ʹ�õ������ռ�  
//-----------------------------------------------------------------------------------------------   
using namespace cv;
using namespace std;
#define SavePatch "D:\\����ռ��\\����\\����02.txt"
#define Patch "D:\\����ռ��\\����\\P (2).JPG"
string SaveImageDirPatch = "D:\\����ռ��\\����\\";
string SaveImageName = "����02 ";
#define WINDOW_NAME "�����򴰿ڡ�" 

//-----------------------------------��ȫ�ֱ����������֡�--------------------------------------  
//      ������ȫ�ֱ�������  
//-----------------------------------------------------------------------------------------------  
double RectA4 = 39601;	//��ֵ���ڼ�¼���ε���ʵ���ֵ
double reactArea = 1800; //�ٶ���������ֵ
Mat g_srcImage, g_dstImage, g_grayImage, g_maskImage, dstmask;//����ԭʼͼ��Ŀ��ͼ���Ҷ�ͼ����ģͼ
int g_nFillMode = 1;//��ˮ����ģʽ
int g_nLowDifference = 47, g_nUpDifference = 46;//�������ֵ���������ֵ
int g_nConnectivity = 4;//��ʾfloodFill������ʶ���Ͱ�λ����ֵͨ
int g_bIsColor = true;//�Ƿ�Ϊ��ɫͼ�ı�ʶ������ֵ
bool g_bUseMask = false;//�Ƿ���ʾ��Ĥ���ڵĲ���ֵ
int g_nNewMaskVal = 255;//�µ����»��Ƶ�����ֵ
int area;	//��ˮ��� ѡ��������ֵ
vector<Point2f> points(4);	//���������꣬���ڿ����Ƿ�����Ż�
Point g_Point;	//����on_MouseHandle()�������������Ĵ洢
bool g_bPoint = false;//on_MouseHandle()�����Ƿ���л���
int cnt = 0;	//��ʼ�����ô�����on_MouseHandle()�Ⱥ�������ʹ��

//-----------------------------------��������������----------------------------------  
//      ���������һЩ������Ϣ  
//---------------------------------------------------------------------------------------------- 
void GetCoordinates(Mat srcImage);
void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawFilledCircle(cv::Mat& img, Point center);

void tf(Point2f q1, Point2f q2, Point2f q3, Point2f q4);
void HSVSplit(Mat image);//V1.0�����˸���
int K_Means(Mat pic);//K-Means�㷨���зָ�
double Contours(Mat src);
void saveImage(string savePatch, Mat image);
//int *ptr =  &area;

//-----------------------------------��ShowHelpText( )������----------------------------------  
//      ���������һЩ������Ϣ  
//----------------------------------------------------------------------------------------------  
static void ShowHelpText()
{
	//���һЩ������Ϣ  
	printf("\n\n\n\t��ӭ������ˮ���ʾ������~\n\n");
	printf("\n\n\t��������˵��: \n\n"
		"\t\t�����ͼ������- ������ˮ������\n"
		"\t\t���̰�����ESC��- �˳�����\n"
		"\t\t���̰�����1��-  �л���ɫͼ/�Ҷ�ͼģʽ\n"
		"\t\t���̰�����2��- ��ʾ/������Ĥ����\n"
		"\t\t���̰�����3��- �ָ�ԭʼͼ��\n"
		"\t\t���̰�����4��- ʹ�ÿշ�Χ����ˮ���\n"
		"\t\t���̰�����5��- ʹ�ý��䡢�̶���Χ����ˮ���\n"
		"\t\t���̰�����6��- ʹ�ý��䡢������Χ����ˮ���\n"
		"\t\t���̰�����7��- ������־���ĵͰ�λʹ��4λ������ģʽ\n"
		"\t\t���̰�����8��- ������־���ĵͰ�λʹ��8λ������ģʽ\n"
	);
}


//-----------------------------------��onMouse( )������--------------------------------------  
//      �����������ϢonMouse�ص�����
//---------------------------------------------------------------------------------------------
static void onMouse(int event, int x, int y, int, void*)
{
	// ��������û�а��£��㷵��
	if (event != EVENT_LBUTTONDOWN)
		return;

	//-------------------��<1>����floodFill����֮ǰ�Ĳ���׼�����֡�---------------
	Point seed = Point(x, y);
	int LowDifference = g_nFillMode == 0 ? 0 : g_nLowDifference;//�շ�Χ����ˮ��䣬��ֵ��Ϊ0��������Ϊȫ�ֵ�g_nLowDifference
	int UpDifference = g_nFillMode == 0 ? 0 : g_nUpDifference;//�շ�Χ����ˮ��䣬��ֵ��Ϊ0��������Ϊȫ�ֵ�g_nUpDifference
	int flags = g_nConnectivity + (g_nNewMaskVal << 8) +
		(g_nFillMode == 1 ? FLOODFILL_FIXED_RANGE : 0);//��ʶ����0~7λΪg_nConnectivity��8~15λΪg_nNewMaskVal����8λ��ֵ��16~23λΪCV_FLOODFILL_FIXED_RANGE����0��

	//�������bgrֵ
	int b = 255;//�������һ��0~255֮���ֵ
	int g = 255;//�������һ��0~255֮���ֵ
	int r = 255;//�������һ��0~255֮���ֵ
	Rect ccomp;//�����ػ��������С�߽��������

	Scalar newVal = g_bIsColor ? Scalar(b, g, r) : Scalar(r*0.299 + g * 0.587 + b * 0.114);//���ػ��������ص���ֵ�����ǲ�ɫͼģʽ��ȡScalar(b, g, r)�����ǻҶ�ͼģʽ��ȡScalar(r*0.299 + g*0.587 + b*0.114)

	Mat dst = g_bIsColor ? g_dstImage : g_grayImage;//Ŀ��ͼ�ĸ�ֵ


	//--------------------��<2>��ʽ����floodFill������-----------------------------
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

	imshow("Ч��ͼ", dst);
	dstmask.copyTo(g_maskImage);
	saveImage(SaveImageDirPatch + SaveImageName + "_Ч��ͼ.jpg", dst);
	//saveImage(SaveImageDirPatch + SaveImageName + "_mask.jpg", dstmask);
	//ptr = &area;
	cout << area << " �����ر��ػ�\n";
}

//-----------------------------------��Contours( )������--------------------------------------  
//      ������ʶ����Ҷ�������������ڼ�������Ҷ���������������ֵ�ķ���
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
	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));//RETR_EXTERNAL:ֻ�������Χ��������������Χ�����ڵ���Χ����������
																						//CHAIN_APPROX_SIMPLE �����������Ĺյ���Ϣ�������������յ㴦�ĵ㱣����contours �����ڣ��յ���յ�֮��ֱ�߶��ϵ���Ϣ�㲻�豣��
																						//CHAIN_APPROX_SIMPLE �����������Ĺյ���Ϣ�������������յ㴦�ĵ㱣����contours �����ڣ��յ���յ�֮��ֱ�߶��ϵ���Ϣ�㲻�豣��

	RNG rng(0);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(dst, contours, i, color, 2, LINE_AA, hierarchy, 0, Point(0, 0));
		//drawContours(
		//InputOutputArray  binImg, // ���ͼ��
		//OutputArrayOfArrays  contours,//  ȫ�����ֵ���������
		//Int contourIdx// ����������
		//const Scalar & color,// ����ʱ����ɫ
		//int  thickness,// �����߿�
		//int  lineType,// �ߵ�����LINE_8
		//InputArray hierarchy,// ���˽ṹͼ
		//int maxlevel,// �������� 0ֻ���Ƶ�ǰ�ģ�1��ʾ���ƻ��Ƶ�ǰ������Ƕ������
		//Point offset = Point()// ����λ�ƣ���ѡ
	}

	imshow("��Ҷ������ͼ", dst);
	saveImage(SaveImageDirPatch + SaveImageName + "_����ͼ.jpg", dst);
	waitKey(0);
	double ConArea = 0.0;
	for (int i = 1; i < contours.size(); i++)
	{
		ConArea += contourArea(contours[i]);
	}
	return ConArea;
}

//-----------------------------------��HSVSplit( )������--------------------------------------  
//      �������ָ�ͼ��ɫ�ȿռ�
//---------------------------------------------------------------------------------------------
void HSVSplit(Mat image) {
	Mat hsvimage, hue;
	imshow("image", image);
	cvtColor(image, hsvimage, COLOR_BGR2HSV); //RGB��HSV��ɫ�ռ��ת��
	//imshow("HSV", hsvimage); //ֱ�Ӱ�HSV����ͼ����RGB��ʽ��ʾ����ʾ������ͼ�����ԭͼ��ͬ
	vector<Mat> hsv;
	split(hsvimage, hsv);//��HSV����ͨ������
	imshow("Sɫ�ȿռ�", hsv.at(1));
	saveImage(SaveImageDirPatch + SaveImageName + "_Sɫ�ȿռ�.jpg", hsv.at(1));

	g_srcImage = hsv.at(1);
	hsv[0] = 0;
	hsv[2] = 0;
	merge(hsv, g_srcImage);

	cvtColor(g_srcImage, g_srcImage, COLOR_BGR2RGB);
	waitKey();
	//destroyWindow("image");
	//destroyWindow("HSV");
	//destroyWindow("Sɫ�ȿռ�");

}

//-----------------------------------��K_Means( )������--------------------------------------  
//      ������ʹ��K_Means����ͼ��ľ��࣬�������õ���2��
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
	saveImage(SaveImageDirPatch + SaveImageName + "_K.jpg", pic);
	waitKey(0);

	return 0;
}

//-----------------------------------��tf( )������--------------------------------------  
//      ������ʹ��͸�ӱ任��ͼ�����У��
//---------------------------------------------------------------------------------------------
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
	/*cout <<"Test"<< squre_pts[0] << squre_pts[1] << squre_pts[2] << squre_pts[3] << endl;*/

	//��points���������Ϊת���������
	for (int i = 0; i < 4; i++) {
		points[i] = squre_pts[i];
		cout << "���point:" << "points[" << i << "]:" << points[i] << endl;
	}
	int offsetSize = 150;
	g_srcImage = Mat::zeros(R.height + offsetSize, R.width + offsetSize, CV_8UC3);
	warpPerspective(src, g_srcImage, transmtx, src.size(), 1, 1);//ԭ��transmtx.size()

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

//-----------------------------------��GetCoordinates( )��DrawFilledCircle������on_MouseHandle����������--------------------------------------  
//      ��������ȡ�������������������tf����������͸�ӱ任��DrawFilledCircle�����������ڻ��㣬�����˵�Ĵ�С������
//---------------------------------------------------------------------------------------------
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
	circle(img, center, 10, Scalar(100, 0, 100), thickness, lineType);

}

//-----------------------------------��saveImage����������--------------------------------------  
//      �����������е�ͼƬ���б���
//---------------------------------------------------------------------------------------------
void saveImage(string savePatch, Mat image) {
	//name д�뱣���·��
	//image �����ͼ��
	imwrite(savePatch, image);
}

//-----------------------------------�����溯����--------------------------------------  
//      ��������ɾ�����С������
//---------------------------------------------------------------------------------------------

//�������������С��������
bool ascendSort(vector<Point> a, vector<Point> b) {
	return a.size() < b.size();

}

//�������������С��������
bool descendSort(vector<Point> a, vector<Point> b) {
	return a.size() > b.size();
}

//��ȥС�����Ĳ���
void ContourRemoval(Mat image, int a) {
	Mat srcImage = image;
	Mat thresholdImage;
	Mat grayImage;
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	threshold(grayImage, thresholdImage, 0, 255, THRESH_OTSU + THRESH_BINARY);
	//Mat resultImage;
	//thresholdImage.copyTo(resultImage);
	vector< vector< Point> > contours;  //���ڱ�������������Ϣ
	vector< vector< Point> > contours2; //���ڱ����������100������
	vector<Point> tempV;				//�ݴ������

	findContours(thresholdImage, contours, RETR_LIST, CHAIN_APPROX_NONE);
	//cv::Mat labels;
	//int N = connectedComponents(resultImage, labels, 8, CV_16U);
	//findContours(labels, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//�������������С������������
	sort(contours.begin(), contours.end(), ascendSort);//��������
	vector<vector<Point> >::iterator itc = contours.begin();
	int i = 0;
	while (itc != contours.end())
	{
		//��������ľ��α߽�
		Rect rect = boundingRect(*itc);
		int x = rect.x;
		int y = rect.y;
		int w = rect.width;
		int h = rect.height;
		//���������ľ��α߽�
		cv::rectangle(srcImage, rect, { 0, 0, 255 }, 1);
		//����ͼƬ
		char str[10];
		//printf(str, "%d.jpg", i++);
		cv::imshow("srcImage", srcImage);
		//imwrite(str, srcImage);
		waitKey(10);//����������Ϊ1000���Կ�����ȥС�������Ĺ���

		if (itc->size() < a)
		{
			//�������������100�����򣬷ŵ�����contours2�У�
			tempV.push_back(Point(x, y));
			tempV.push_back(Point(x, y + h));
			tempV.push_back(Point(x + w, y + h));
			tempV.push_back(Point(x + w, y));
			contours2.push_back(tempV);
			/*Ҳ����ֱ���ã�contours2.push_back(*itc);���������5�����*/
			//contours2.push_back(*itc);

			//ɾ�������������100�����򣬼��ú�ɫ��������������100������
			cv::drawContours(srcImage, contours2, -1, Scalar(255, 255, 255), FILLED);
		}
		//����ͼƬ
		//printf(str, "%d.jpg", i++);
		cv::imshow("srcImage", srcImage);
		//imwrite(str, srcImage);
		cv::waitKey(1);
		tempV.clear();
		++itc;
	}
	waitKey(0);
}

//-----------------------------------��main( )������--------------------------------------------  
//      ����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ  
//-----------------------------------------------------------------------------------------------  
int main(int argc, char** argv)
{
	ofstream out;
	out.open(SavePatch, ios::trunc);
	//�ı�console������ɫ  
	system("color 2F");

	//����ԭͼ
	g_srcImage = imread(Patch, 1);

	if (!g_srcImage.data) { printf("Oh��no����ȡͼƬimage0����~�� \n"); return false; }

	//��ʾ��������
	ShowHelpText();

	g_srcImage.copyTo(g_dstImage);//����Դͼ��Ŀ��ͼ
	cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);//ת����ͨ����image0���Ҷ�ͼ
	g_maskImage.create(g_srcImage.rows + 2, g_srcImage.cols + 2, CV_8UC1);//����image0�ĳߴ�����ʼ����Ĥmask

	namedWindow("Ч��ͼ", WINDOW_AUTOSIZE);

	//����Trackbar
	createTrackbar("�������ֵ", "Ч��ͼ", &g_nLowDifference, 255, 0);
	createTrackbar("�������ֵ", "Ч��ͼ", &g_nUpDifference, 255, 0);

	//���ص�����
	setMouseCallback("Ч��ͼ", onMouse, 0);

	//ѭ����ѯ����
	while (1)
	{
		//����ʾЧ��ͼ
		imshow("Ч��ͼ", g_bIsColor ? g_dstImage : g_grayImage);

		//��ȡ���̰���
		int c = waitKey(0);
		//�ж�ESC�Ƿ��£������±��˳�
		if ((c & 255) == 27)
		{
			cout << "�����˳�...........\n";
			dstmask.copyTo(g_maskImage);
			break;
		}

		//���ݰ����Ĳ�ͬ�����и��ֲ���
		switch ((char)c)
		{
			//������̡�1�������£�Ч��ͼ���ڻҶ�ͼ����ɫͼ֮�以��
		case '1':
			if (g_bIsColor)//��ԭ��Ϊ��ɫ��תΪ�Ҷ�ͼ�����ҽ���Ĥmask����Ԫ������Ϊ0
			{
				cout << "���̡�1�������£��л���ɫ/�Ҷ�ģʽ����ǰ����Ϊ������ɫģʽ���л�Ϊ���Ҷ�ģʽ��\n";
				cvtColor(g_srcImage, g_grayImage, COLOR_BGR2GRAY);
				g_maskImage = Scalar::all(0);	//��mask����Ԫ������Ϊ0
				g_bIsColor = false;	//����ʶ����Ϊfalse����ʾ��ǰͼ��Ϊ��ɫ�����ǻҶ�
			}
			else//��ԭ��Ϊ�Ҷ�ͼ���㽫ԭ���Ĳ�ͼimage0�ٴο�����image�����ҽ���Ĥmask����Ԫ������Ϊ0
			{
				cout << "���̡�1�������£��л���ɫ/�Ҷ�ģʽ����ǰ����Ϊ������ɫģʽ���л�Ϊ���Ҷ�ģʽ��\n";
				g_srcImage.copyTo(g_dstImage);
				g_maskImage = Scalar::all(0);
				g_bIsColor = true;//����ʶ����Ϊtrue����ʾ��ǰͼ��ģʽΪ��ɫ
			}
			break;
			//������̰�����2�������£���ʾ/������Ĥ����
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
			//������̰�����3�������£��ָ�ԭʼͼ��
		case '3':
			cout << "������3�������£��ָ�ԭʼͼ��\n";
			g_srcImage.copyTo(g_dstImage);
			cvtColor(g_dstImage, g_grayImage, COLOR_BGR2GRAY);
			g_maskImage = Scalar::all(0);
			break;
			//������̰�����4�������£�ʹ�ÿշ�Χ����ˮ���
		case '4':
			cout << "������4�������£�ʹ�ÿշ�Χ����ˮ���\n";
			g_nFillMode = 0;
			break;
			//������̰�����5�������£�ʹ�ý��䡢�̶���Χ����ˮ���
		case '5':
			cout << "������5�������£�ʹ�ý��䡢�̶���Χ����ˮ���\n";
			g_nFillMode = 1;
			break;
			//������̰�����6�������£�ʹ�ý��䡢������Χ����ˮ���
		case '6':
			cout << "������6�������£�ʹ�ý��䡢������Χ����ˮ���\n";
			g_nFillMode = 2;
			break;
			//������̰�����7�������£�������־���ĵͰ�λʹ��4λ������ģʽ
		case '7':
			cout << "������7�������£�������־���ĵͰ�λʹ��4λ������ģʽ\n";
			g_nConnectivity = 4;
			break;
			//������̰�����8�������£�������־���ĵͰ�λʹ��8λ������ģʽ
		case '8':
			cout << "������8�������£�������־���ĵͰ�λʹ��8λ������ģʽ\n";
			g_nConnectivity = 8;
			break;
		}
	}

	////��ȡ�������������������tf����������͸�ӱ任
	//GetCoordinates(g_srcImage);
	//waitKey(0);
	//destroyWindow(WINDOW_NAME);
	////��ͼ�����͸�ӱ任
	//tf(points[0], points[1], points[2], points[3]);
	////������ε����ֵ�����������Ρ�
	//double reactX = abs(points[1].x - points[0].x);
	//double reactY = abs(points[3].y - points[1].y);
	//double reactArea = reactX * reactY;
	//cout << "����ռ���������Ϊ:" << reactArea << endl;
	//out << "����ռ���������Ϊ:" << reactArea << endl;

	Mat dstRect = g_srcImage.clone();
	rectangle(dstRect, points[0], points[2], (0, 0, 255), 5);	//��������
	imshow("ʶ��ľ���", dstRect);
	saveImage(SaveImageDirPatch + SaveImageName + "_ʶ��ľ���.jpg", dstRect);

	waitKey(0);
	destroyWindow("ʶ��ľ���");
	//ʹ��HSV����ͼ��ָ�
	HSVSplit(g_srcImage);
	//K-Means���ж�ͼ��Ķ�ֵ���ͷ���
	K_Means(g_srcImage);
	//��ʴ����
	Mat element = getStructuringElement(MORPH_RECT, Size(2, 5));//�����
	erode(g_srcImage, g_srcImage, element);
	imshow("��ֵ��", g_srcImage);
	//��ȥС�������֣���������ֵΪ300
	ContourRemoval(g_srcImage, 50);
	saveImage(SaveImageDirPatch + SaveImageName + "_��ֵ��.jpg", g_srcImage);

	waitKey(0);
	destroyAllWindows();
	//ʶ����Ҷ�������������ڼ�������Ҷ���������������ֵ�ķ���	
	double ConArea = Contours(g_srcImage);
	cout << "��Ҷ������Ϊ��" << ConArea << "\n" << "��Ҷ��������Ϊ��" << (ConArea - area) << "\n" << "��Ҷ�Ĳ����ı���Ϊ��" << (ConArea - area) / ConArea << endl;
	out << "��Ҷ������Ϊ��" << ConArea << "\n" << "��Ҷ��������Ϊ��" << (ConArea - area) << "\n" << "��Ҷ�Ĳ����ı���Ϊ��" << (ConArea - area) / ConArea << endl;
	//������Ҷ����ʵ���
	double lArea = RectA4 * ConArea / reactArea;
	double pArea = RectA4 * (ConArea - area) / reactArea;
	cout << "��Ҷ����ʵ���Ϊ��" << lArea << "\n" << "��Ҷ������ʵ���Ϊ��" << pArea << endl;
	out << "��Ҷ����ʵ���Ϊ��" << lArea << "\n" << "��Ҷ������ʵ���Ϊ��" << pArea << endl;
	waitKey(0);


	out.close();

	return 0;
}
