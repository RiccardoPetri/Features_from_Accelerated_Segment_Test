#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <circleHandler.h>

#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;


CircumferenceInfo::CircumferenceInfo(int firstDarkPixel, int lastDarkPixel, int firstBrightPixel, int lastBrightPixel, int darkPixelCounter, int brightPixelCounter) {

    startDarkAngle = firstDarkPixel;			
    endDarkAngle = lastDarkPixel;
    startBrightAngle = firstBrightPixel;
    endBrightAngle = lastBrightPixel;
    darkConsecutive = darkPixelCounter;
    brightConsecutive = brightPixelCounter;
}


vector<Point> getCircumference(int x0, int y0, int radius){ //According to the radius, the points belonging to the circumference, created from the center pixel p,
    							    //are added to the vector "circumferencePoints" in an anticlockwise direction

    vector<Point> circumferencePoints;
    switch (radius) {	
    
    case 1:
        circumferencePoints.push_back(Point(x0,y0-1)); 			
        circumferencePoints.push_back(Point(x0+1,y0-1)); 		
        circumferencePoints.push_back(Point(x0+1,y0)); 			//	      0	0 0
        circumferencePoints.push_back(Point(x0+1,y0+1)); 		//	      0 p 0   
        circumferencePoints.push_back(Point(x0,y0+1));  		//	      0	0 0		    
        circumferencePoints.push_back(Point(x0-1,y0+1)); 				  
        circumferencePoints.push_back(Point(x0-1,y0)); 					
        circumferencePoints.push_back(Point(x0-1,y0+1)); 			
        return circumferencePoints;
    case 2:
        circumferencePoints.push_back(Point(x0,y0-2)); 			
        circumferencePoints.push_back(Point(x0+1,y0-2)); 		
        circumferencePoints.push_back(Point(x0+2,y0-1)); 				
        circumferencePoints.push_back(Point(x0+2,y0)); 			//	       000
        circumferencePoints.push_back(Point(x0+2,y0+1)); 		//	      0   0
        circumferencePoints.push_back(Point(x0+1,y0+2)); 		//	      0	p 0
        circumferencePoints.push_back(Point(x0,y0+2)); 			//	      0   0
        circumferencePoints.push_back(Point(x0-1,y0+2)); 		//	       000
        circumferencePoints.push_back(Point(x0-2,y0+1)); 
        circumferencePoints.push_back(Point(x0-2,y0)); 
        circumferencePoints.push_back(Point(x0-2,y0-1)); 
        circumferencePoints.push_back(Point(x0-1,y0-2));         	  
        return circumferencePoints;
    case 3:
        circumferencePoints.push_back(Point(x0,y0-3)); 			
        circumferencePoints.push_back(Point(x0+1,y0-3)); 
        circumferencePoints.push_back(Point(x0+2,y0-2)); 
        circumferencePoints.push_back(Point(x0+3,y0-1)); 
        circumferencePoints.push_back(Point(x0+3,y0));                  //            000
        circumferencePoints.push_back(Point(x0+3,y0+1));                //           0   0
        circumferencePoints.push_back(Point(x0+2,y0+2));                //          0     0
        circumferencePoints.push_back(Point(x0+1,y0+3)); 		//	    0  p  0
        circumferencePoints.push_back(Point(x0,y0+3));                  //          0     0
        circumferencePoints.push_back(Point(x0-1,y0+3)); 		//	     0   0
        circumferencePoints.push_back(Point(x0-2,y0+2)); 		//	      000
        circumferencePoints.push_back(Point(x0-3,y0+1));
        circumferencePoints.push_back(Point(x0-3,y0)); 
        circumferencePoints.push_back(Point(x0-3,y0-1)); 
        circumferencePoints.push_back(Point(x0-2,y0-2)); 
        circumferencePoints.push_back(Point(x0-1,y0-3)); 		
        return circumferencePoints;
    case 4:
        circumferencePoints.push_back(Point(x0,y0-4)); 
        circumferencePoints.push_back(Point(x0+1,y0-4)); 
        circumferencePoints.push_back(Point(x0+2,y0-3)); 		 
        circumferencePoints.push_back(Point(x0+3,y0-2)); 		 
        circumferencePoints.push_back(Point(x0+4,y0-1)); 		 			
        circumferencePoints.push_back(Point(x0+4,y0)); 			 //	       000
        circumferencePoints.push_back(Point(x0+4,y0+1)); 		 //	      0   0
        circumferencePoints.push_back(Point(x0+3,y0+2)); 		 //	     0     0
        circumferencePoints.push_back(Point(x0+2,y0+3)); 		 //	    0       0
        circumferencePoints.push_back(Point(x0+1,y0+4)); 		 //	    0   p   0
        circumferencePoints.push_back(Point(x0,y0+4)); 		 	 //	    0       0
        circumferencePoints.push_back(Point(x0-1,y0+4)); 		 //	     0     0
        circumferencePoints.push_back(Point(x0-2,y0+3)); 		 //	      0   0
        circumferencePoints.push_back(Point(x0-3,y0+2)); 		 //	       000
        circumferencePoints.push_back(Point(x0-4,y0+1)); 		 
        circumferencePoints.push_back(Point(x0-4,y0)); 		 	
        circumferencePoints.push_back(Point(x0-4,y0-1)); 		 
        circumferencePoints.push_back(Point(x0-3,y0-2)); 		 
        circumferencePoints.push_back(Point(x0-2,y0-3)); 
        circumferencePoints.push_back(Point(x0-1,y0-4)); 
        return circumferencePoints;
    case 5:
        circumferencePoints.push_back(Point(x0,y0-5)); 
        circumferencePoints.push_back(Point(x0+1,y0-5)); 
        circumferencePoints.push_back(Point(x0+2,y0-4)); 		 
        circumferencePoints.push_back(Point(x0+3,y0-3)); 		 //
        circumferencePoints.push_back(Point(x0+4,y0-2)); 		 //			
        circumferencePoints.push_back(Point(x0+5,y0-1)); 		 //	       000
        circumferencePoints.push_back(Point(x0+5,y0)); 			 //	      0   0
        circumferencePoints.push_back(Point(x0+5,y0+1)); 		 //	     0     0
        circumferencePoints.push_back(Point(x0+4,y0+2)); 		 //	    0       0
        circumferencePoints.push_back(Point(x0+3,y0+3)); 		 //	   0         0
        circumferencePoints.push_back(Point(x0+2,y0+4)); 		 //        0    p    0   
        circumferencePoints.push_back(Point(x0+1,y0+5)); 		 //	   0         0
        circumferencePoints.push_back(Point(x0,y0+5)); 			 //	    0	    0
        circumferencePoints.push_back(Point(x0-1,y0+5)); 		 //	     0     0
        circumferencePoints.push_back(Point(x0-2,y0+4)); 		 //	      0   0
        circumferencePoints.push_back(Point(x0-3,y0+3)); 		 //            000
        circumferencePoints.push_back(Point(x0-4,y0+2)); 		 //
        circumferencePoints.push_back(Point(x0-5,y0+1)); 		 //
        circumferencePoints.push_back(Point(x0-5,y0)); 
        circumferencePoints.push_back(Point(x0-5,y0-1)); 
	circumferencePoints.push_back(Point(x0-4,y0-2)); 
	circumferencePoints.push_back(Point(x0-3,y0-3)); 
	circumferencePoints.push_back(Point(x0-2,y0-4)); 
	circumferencePoints.push_back(Point(x0-1,y0-5)); 
        return circumferencePoints;
    default:
        cerr<<"ERROR: input radius not declare"<<endl;
        exit(EXIT_FAILURE);
    }




}


int getAngle(int index, int radius)			//The function returns the angle given the index and the radius.
							//Using bigger radii we can find more angles having more precision.
{
    switch(radius) {
    case 2:
        switch(index) {
        case 0: return 0;
        case 1: return 30;
        case 2: return 60;
        case 3: return 90;
        case 4: return 120;
        case 5: return 150;
        case 6: return 180;
        case 7: return 210;
        case 8: return 240;
        case 9: return 270;
        case 10: return 300;
        case 11: return 330;
        default:
            cerr<<"ERROR: invalid index with radius 2"<<endl;
            exit(EXIT_FAILURE);
        }
        break;
   case 3:
        switch(index) {
        case 0: return 0;
        case 1: return 22.5;
        case 2: return 45;
        case 3: return 67.5;
        case 4: return 90;
        case 5: return 112.5;
        case 6: return 135;
        case 7: return 157.5;
        case 8: return 180;
        case 9: return 202.5;
        case 10: return 225;
        case 11: return 247.5;
        case 12: return 270;
        case 13: return 292.5;
        case 14: return 315;
        case 15: return 337.5;
        default:
            cerr<<"ERROR: invalid index with radius 3"<<endl;
            exit(EXIT_FAILURE);
        }
        break;
    default:
        cerr<<"ERROR: input radius not declare"<<endl;
        exit(EXIT_FAILURE);
    }

    return -1;
}

