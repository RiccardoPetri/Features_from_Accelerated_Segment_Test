#ifndef _CORNERDETECTION_H_
#define _CORNERDETECTION_H_
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "circleHandler.h"

#define THRESHOLD 55 		// Intensity value of a pixel to detect a possible corner ("pixel_p_intensity > comparisonPixelIntensity + THRESHOLD")
#define DARKER_PIXEL 1		// Flag to demonstrate that there is a darker pixel on the circumference in the last iteration
#define SIMILAR_PIXEL 2		// Flag to demonstrate that pixels are similar in the last iteration	
#define BRIGHTER_PIXEL 3	// Flag to demonstrate that there is a brighter pixel on the circumference in the last iteration

using namespace cv;
using namespace std;

class cornerDetection {

    vector<Point> keyPoints; //Vector containing the points to be marked
    Mat convertPixelToGreyScale(const Mat& image); //Function to convert the root image pixel intensity into grey scale
    int getGreyScalePixelIntensity(const Mat& image, const int& x, const int& y); //Function to return the intensity in grey scale of a given pixel
    CircumferenceInfo calculateCircumferenceInfo(const Mat& image, const int x, const int y, const int r); /* Function to return circumference pixels information 
    													   given the image, the coordinates of pixel p and the radius */
    int size; 
    int count = 0;													   
    int calculateFacingAngle(const CircumferenceInfo& cd); //Function to determine the angle which the consecutive brighter or darker pixel is facing
    bool pixelScan(const Mat& image, const int x, const int y); //Function to scan pixels in order to determine if they are corners
  public:
    cornerDetection();
    Mat getGreyScaleImage(const Mat& image);
    vector<Point> pushKeyPointsInVector(const Mat& portionMatrix, int rank, int portionRow);
    void findKeyPoints(const Mat& image); //Function to determine points to be marked on image as corners
    void showKeyPoints(Mat& image, vector<Point> points); //Function to draw circles in corrispondence of corners
};

#endif	
