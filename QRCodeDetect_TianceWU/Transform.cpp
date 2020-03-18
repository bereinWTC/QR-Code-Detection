//env：VS2019 + opencv 4.1.2

#include<opencv2\opencv.hpp>
#include <iostream>     
#include<math.h>


using namespace std;
using namespace cv;

struct index
{
	int index;
	int cont1;
	int cont2;
};
Point2f GetContourCenter(vector<vector<Point>> contours, int index)
{
	Point2f p;
	int contourlength = contours[index].size();
	double avg_px = 0, avg_py = 0;
	for (int i = 0; i < contourlength; i++)
	{
		avg_px += contours[index][i].x;
		avg_py += contours[index][i].y;
	}
	p.x = int(avg_px / contourlength);
	p.y = int(avg_py / contourlength);
	return p;
}

//已知一个矩形三个点。求中心和第四个点
vector<Point2f> Getmaximumdistance(Point2f point1, Point2f point2, Point2f point3) {
	vector<Point2f> fois;
	vector<Point2f>target;
	Point2f center, fourth,first;
	float distance1, distance2, distance3, dismax;
	distance1 = sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
	distance2 = sqrt(pow(point1.x - point3.x, 2) + pow(point1.y - point3.y, 2));
	distance3 = sqrt(pow(point2.x - point3.x, 2) + pow(point2.y - point3.y, 2));
	dismax = max(max(distance1, distance2), distance3);
	if (dismax == distance1) { fois.push_back(point1); fois.push_back(point2); first = point3; }
	else if (dismax == distance2) { fois.push_back(point1); fois.push_back(point3); first=point2;
	}
	else { fois.push_back(point2); fois.push_back(point3); first = point1;
	}
	center = Point2f((fois[0].x + fois[1].x) / 2, (fois[0].y + fois[1].y) / 2);
	fourth = Point2f((4 * center.x - point1.x-point2.x-point3.x), (4 * center.y - point1.y-point2.y-point3.y));
	target.push_back(center);
	target.push_back(fourth);
	return target;

}

int main() {
	Scalar red = Scalar(1, 1, 255);
	//图像预处理
	Mat imageSource = imread("14_WU_Tiance.bmp", 0);
	Mat image;
	imageSource.copyTo(image);
	threshold(image, image, 100, 255, THRESH_BINARY);  //二值化
	imshow("BW", image);
	Mat element = getStructuringElement(2, Size(7, 7));	 //膨胀腐蚀核
	
	int rounds = 15;
	for (int i = 0; i < rounds; i++) {
		erode(image, image, element);
		i++;
	}

	imshow("erosions", image);
	imwrite("erosion.bmp", image);
	Mat image1;
	erode(image, image1, element);

	image1 = image - image1;
	//imshow("borders", image1);


	vector<Vec2f>lines;
	// 霍夫变换拉线 拉到最重要的几条线
	HoughLines(image1, lines, 1, CV_PI / 150, 250, 0, 0); 
	Mat DrawLine = Mat::zeros(image1.size(), CV_8UC1);
	for (int i = 0; i < lines.size(); i++)
	{

		float rho = lines[i][0];
		float theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;

		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * a);
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * a);

		line(DrawLine, pt1, pt2, Scalar(255), 1, 16);//老旧版本的CV_AA = 16.现在不通了
	}

	//imshow("lines", DrawLine);
	vector<Point2f>corners;
	goodFeaturesToTrack(DrawLine, corners, 4, 0.1, 10, Mat()); //利用之前霍夫变换拉的线找到四个顶点

	for (int i = 0; i < corners.size(); i++)
	{
		circle(DrawLine, corners[i], 3, Scalar(255), 3);
		circle(imageSource, corners[i], 3, red, 3);
		//P1[i] = corners[i];
	}
	//imshow("points", DrawLine);
	imwrite("lineandcorner.bmp", DrawLine);

	//定义四个基础的点
	float xdiffer,ydiffer;
	xdiffer = corners[1].x - corners[0].x;
	ydiffer = corners[2].y - corners[0].y;

	vector<Point2f>targets;
	targets.push_back(corners[0]);
	targets.push_back(Point2f(corners[0].x+xdiffer, corners[0].y));
	targets.push_back(Point2f(corners[0].x, corners[0].y + ydiffer));
	targets.push_back(Point2f(corners[0].x + xdiffer, corners[0].y + ydiffer));

	//拧图 拧到正面
	Mat elementTransf;
	Mat ImageOut, ImageCut;
	elementTransf = getPerspectiveTransform(corners, targets);
	warpPerspective(imageSource, ImageOut, elementTransf, imageSource.size());
	imwrite("transformed.bmp", ImageOut);
	//裁剪并调整尺寸
	int min_x, min_y, max_x, max_y;
	min_x = min(min(targets[0].x, targets[1].x), min(targets[2].x, targets[3].x));
	max_x = max(max(targets[0].x, targets[1].x), max(targets[2].x, targets[3].x));
	min_y = min(min(targets[0].y, targets[1].y), min(targets[2].y, targets[3].y));
	max_y = max(max(targets[0].y, targets[1].y), max(targets[2].y, targets[3].y));
	Rect m_select;
	
	m_select= Rect(min_x, min_y, max_x-min_x, max_y-min_y);
	ImageCut = ImageOut(m_select);
	imwrite("cut.bmp", ImageCut);
	imshow("cut.bmp", ImageCut);


	namedWindow("Source Window", 0);
	imshow("Source Window", imageSource);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;



	//计算轮廓
	findContours(ImageCut, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
	Mat imgContours = Mat::zeros(ImageCut.size(), CV_8UC3);
	drawContours(imgContours, contours, -1, CV_RGB(255, 255, 255), 1, 8);
	imwrite("contour.bmp", imgContours);
	//找三个回
	vector<index> candidate;
	vector<index>::iterator it;
	index tmp;
	int tmp1, tmp2;
	for (int i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][2] == -1)
			continue;
		else
			tmp1 = hierarchy[i][2];
		if (hierarchy[tmp1][2] == -1)
			continue;
		else
		{
			tmp2 = hierarchy[tmp1][2];
			tmp.index = i;
			tmp.cont1 = tmp1;
			tmp.cont2 = tmp2;
			candidate.push_back(tmp);
		}
	}
	//通过三个回找到中心和第四个点
	vector<Point2f>Candidates,twopoints;

	int i;
	for (it = candidate.begin(), i = 0; it != candidate.end(); it++, i++)
	{
		Point2f point = GetContourCenter(contours, it->index);
		Candidates.push_back(point);
	}
	Rect rectii = Rect(35, 25, 605, 580);
	ImageCut = ImageCut(rectii);
	resize(ImageCut, ImageCut, Size(420, 420));
	twopoints = Getmaximumdistance(Candidates[0], Candidates[1], Candidates[2]);
	float judge0, judge1;
	judge0 = twopoints[1].x - twopoints[0].x;
	judge1 = twopoints[1].y - twopoints[0].y;
	//把第四个点旋转到右下角。可以通过第四个点减中心的坐标差判定要转多少度，前面变换之后的图一定是方的，所以只有90 180 270 三种情况
	Mat ImageTarget,R;
	if(judge0>0 && judge1>0){//第四个点在右下角
		R = getRotationMatrix2D(twopoints[0], 0, 1.0);
	}
	else if (judge0 > 0 && judge1 < 0) {//第四个点在右上角
		R = getRotationMatrix2D(twopoints[0], 90, 1.0);
	}
	else if (judge0 < 0 && judge1 < 0) {//第四个点在左上角
		R = getRotationMatrix2D(twopoints[0], 180, 1.0);
	}
	else {//第四个点在左下角
		R = getRotationMatrix2D(twopoints[0], 270, 1.0);
	}

	warpAffine(ImageCut, ImageTarget, R, ImageCut.size());
	
	circle(ImageTarget, twopoints[1], 10, red, 2);
	imshow("Target", ImageTarget);
	resize(ImageTarget, ImageTarget, Size(42, 42));
	threshold(ImageTarget, ImageTarget, 112, 255, THRESH_BINARY);
	imwrite("final.bmp", ImageTarget);
	
	waitKey();

	return 0;
}
