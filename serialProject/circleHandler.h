#ifndef _CIRCLEHANDLER_H_		/*
					#ifndef checks whether the given token has been #defined earlier in the file or in an included file; if not, 
					it includes the code between it and the closing #else or, if no #else is present, #endif statement. #ifndef 
  					is often used to make header files idempotent by defining a token once the file has been included and checking
					that the token was not set at the top of that file.
					*/
	
#define _CIRCLEHANDLER_H_		

#include <opencv2/core/core.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct CircumferenceInfo {					//Information of the circumference around the candidate pixel

    CircumferenceInfo(int startDarkPixel, int lastDarkPixel, int firstBrightPixel, int lastBrightPixel, int darkPixelCounter, int brightPixelCounter);
    int startDarkAngle;						//
    int endDarkAngle;						//
    int startBrightAngle;					// Values representing the biggest dark angle and bright angle
    int endBrightAngle;						//
    int darkConsecutive;					//
    int brightConsecutive;					//
};

vector<Point> getCircumference(int x0, int y0, int radius);	//This method will return the vector of Points of the circumference elements given the 
								//coordinates x0 y0 of the center pixel p and the radius

int getAngle(int index, int radius);			        //This method will return the angle given the index and the radius 
	  
#endif


