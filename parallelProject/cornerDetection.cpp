#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cornerDetection.h>
#include <mpi.h>

#include <vector>



using namespace cv;
using namespace std;


//Mat Input: pass a copy of Input's header. Its header will not be changed outside of this function, but can be changed within the function

//Const Mat Input: pass a copy of Input's header. Its header will not be changed outside of or within the function. 

//Const Mat& Input: pass a reference of Input's header. Guarantees that Input's header will not be changed outside of or within the function

//Mat& Input: pass a reference of Input's header. Changes to Input's header happen outside of and within the function



cornerDetection::cornerDetection(){}



Mat cornerDetection::convertPixelToGreyScale(const Mat& image) { //This function return the matrix of the pixels after converting it into greyscale

    Mat greyScaleMatrix(image);	//takes a ref called greyScaleMatrix (image is already a MAT, so it is already splitted in 3 contigous vertical r-g-b value for each pixel, 
				//but here we want to do an avarage!!

    Mat matrice(greyScaleMatrix.rows, greyScaleMatrix.cols, CV_32S);
   

    for(int y=0; y<greyScaleMatrix.rows; y++) { //loop scrolling pixels of the matrix along the row
        Vec3b* ptr = greyScaleMatrix.ptr<Vec3b>(y);
	int* ptrr = matrice.ptr<int>(y);
        for(int x=0; x<greyScaleMatrix.cols; x++) { //Scrolling pixels on row Y changing columns
	    int averagePixelIntensity = (ptr[x][0]+ptr[x][1]+ptr[x][2])/3; //Pixel ptr[x] mean of his 3 value to obtain the conversion to grey scale
            ptr[x] = Vec3b(averagePixelIntensity,averagePixelIntensity,averagePixelIntensity); //Vec3b represents a triplet of uchar value (integer from 0 and 255).
	    ptrr[x] = averagePixelIntensity;
        }	 
    }
    return matrice;
}



int cornerDetection::getGreyScalePixelIntensity(const Mat& image, const int& x, const int& y) { // Function to obtain pixel intensity from coordinates
   
    int pixelIntensityValue=-1;
    const int* ptr = image.ptr<int>(y);
    pixelIntensityValue = ptr[x]; //Gives to greyScaleMatrix the first channel value (equal to the other 2) to return an integer that represents the pixel(X,Y) grey level intensity
    return pixelIntensityValue;
}


CircumferenceInfo cornerDetection::calculateCircumferenceInfo(const Mat& image, const int x, const int y, const int r) { //Function used to search on the circumference the longest consecutive 
															 //BRIGHTER_PIXEL or DARKER_PIXEL pixels which means a corner
      int pixel_p_intensity = getGreyScalePixelIntensity(image,x,y);


    vector<Point> circumferenceValues = getCircumference(x, y, r); //Vector of points, so each value has x and y

    int circumferenceVectorDimension = (int)circumferenceValues.size();

    int darkConsecutive=0, brightConsecutive=0, consecutive=0;
    int startDarkAngle=-1, endDarkAngle=-1;
    int startBrightAngle=-1, endBrightAngle=-1;
    int angle=-1;

    unsigned int intensity=0;
    
    for(int i=0; i<2*circumferenceVectorDimension; i++) {
        int comparisonPixel_x = circumferenceValues[i%circumferenceVectorDimension].x;
        int comparisonPixel_y = circumferenceValues[i%circumferenceVectorDimension].y;
        int comparisonPixelIntensity = getGreyScalePixelIntensity(image, comparisonPixel_x, comparisonPixel_y);

	
        if (pixel_p_intensity > comparisonPixelIntensity + THRESHOLD) {	//Case in which p has a brighter intensity than the pixel considered on the circumference
            if (i==0) {
                angle=getAngle(i%circumferenceVectorDimension,r); //% Operator providing the remainder of the division (1/8 -> getAngle("case 1", radius)) 
            } else if (intensity == DARKER_PIXEL) {
                consecutive++; //If the last circumference pixel "comparisonPixelIntensity" is a DARKER_PIXEL and now we are into this condition, we have two consecutive dark pixels -> consecutive++
            } else {
                if (intensity == BRIGHTER_PIXEL && consecutive > brightConsecutive) { //If we enter in this condition, it means that we ran out of consecutive pixels with "DARKER_PIXEL"
										      //("brightConsecutive" stores the number of the consecutive bright pixels of the circumference of the last 
										      //analysis)
                    endBrightAngle = getAngle(i%circumferenceVectorDimension,r);
                    startBrightAngle = angle;
                    brightConsecutive = consecutive;
                }
                angle=getAngle(i%circumferenceVectorDimension,r); //For the following "angle discover" we set this position as the start of the new angle
                consecutive=1; 					  //We reinitialize the counter
            }
            intensity = DARKER_PIXEL; //This will be used for the next iteration
        } else if (pixel_p_intensity < comparisonPixelIntensity - THRESHOLD) { //Case in which p has a darker intensity than the pixel considered on the circumference
            if (i==0) {
                angle=getAngle(i%circumferenceVectorDimension,r);
            } else if (intensity == BRIGHTER_PIXEL) {
                consecutive++;
            } else {
                if (intensity == DARKER_PIXEL && consecutive > darkConsecutive) { //If we enter in this condition, it means that we ran out of consecutive pixels with "BRIGHTER_PIXEL"
										  //("darkConsecutive" stores the number of the consecutive dark pixels of the circumference of the last 
										  //analysis)
                    endDarkAngle = getAngle(i%circumferenceVectorDimension,r);
                    startDarkAngle = angle;
                    darkConsecutive = consecutive;
                }
                angle=getAngle(i%circumferenceVectorDimension,r);  //For the following "angle discover" we set this position as the start of the new angle
                consecutive=1;		                           //We reinitialize the counter
            }
            intensity = BRIGHTER_PIXEL;
        } else { 						   //Case in which p has a similar intensity than the pixel considered on the circumference
            if (intensity == BRIGHTER_PIXEL) {
                if (consecutive > brightConsecutive) {
                    endBrightAngle = getAngle(i%circumferenceVectorDimension,r);
                    startBrightAngle = angle;
                    brightConsecutive = consecutive;
                }
            } else if (intensity == DARKER_PIXEL) {
                if (consecutive > darkConsecutive) {
                    endDarkAngle = getAngle(i%circumferenceVectorDimension,r);
                    startDarkAngle = angle;
                    darkConsecutive = consecutive;
                }
            }
            intensity = SIMILAR_PIXEL;
        }
    }
    
    CircumferenceInfo cInfo(startDarkAngle, endDarkAngle, startBrightAngle, endBrightAngle, darkConsecutive, brightConsecutive);
    return cInfo;
}


int cornerDetection::calculateFacingAngle(const CircumferenceInfo& cInfo) { //Function used to return the angle opposite to the consecutive darkness
									   // or brightness 

    int meanAngle = -1;
    if (cInfo.darkConsecutive > cInfo.brightConsecutive) {
        meanAngle = (cInfo.startDarkAngle + cInfo.endDarkAngle) / 2;
    } else {
        meanAngle = (cInfo.startBrightAngle + cInfo.endBrightAngle) / 2;
    }
    return meanAngle;
}



bool cornerDetection::noiseEliminator(const Mat& image, const int x, const int y) { //Function used to eliminate found corners that are characterized by noise
										    //They can be recognized by facing angles, infact noise angles have different directions

    CircumferenceInfo cInfo2 = calculateCircumferenceInfo(image, x, y, 2);
    CircumferenceInfo cInfo3 = calculateCircumferenceInfo(image, x, y, 3);

    int facingAngle2 = calculateFacingAngle(cInfo2);
    int facingAngle3 = calculateFacingAngle(cInfo3);



    int delta23 = abs(facingAngle2-facingAngle3); //Difference between angles to verify if they have noise


    if (delta23 < ANGULAR_THRESHOLD) { //Check if angles have similar directions, if true they are mantained
        return true;
    } else {
        return false;
    }
}


bool cornerDetection::pixelScan(const Mat& image, const int x, const int y) { //Function that represents the basic FAST algorithm activity

    CircumferenceInfo cInfo = calculateCircumferenceInfo(image, x, y, 3);
    int darkConsecutive = cInfo.darkConsecutive;
    int brightConsecutive = cInfo.brightConsecutive;
    if (darkConsecutive>11 || brightConsecutive>11) {
        return true;
    }
    return false;
}




Mat cornerDetection::getGreyScaleImage(const Mat& image) { //Function that uses the angle detection functions in order to determine the keypoints to draw
    
    printf("convertPixelToGreyScale() invoked\n");
    
    Mat greyScaleImage = convertPixelToGreyScale(image);
    int numbRows = greyScaleImage.rows;
    int numbCols = greyScaleImage.cols;
    printf("matrix's rows are:  %d\n",numbRows);
    printf("matrix's columns are:  %d\n",greyScaleImage.cols);
    printf("convertPixelToGreyScale() has finished its task\nScanning each pixel with pixelScan() and for keyPoints push_back()\n");		

    MPI_Bcast(&numbCols, 1, MPI_INT, 0, MPI_COMM_WORLD);       //Broadcasting the number of columns of the image matrix to all the slaves

       
    return greyScaleImage;
}




vector<Point> cornerDetection::pushKeyPointsInVector(const Mat& portionMatrix, int rank, int portionRow) { //Function that uses the angle detection functions in order to determine the keypoints to draw
    
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    int value = 0;
 
   
    if(rank!=0){
    	value = ((rank*(portionRow)));			
	value = value-6;			//Shifts point obtained using overlap
    }


    for(int y=3; y<portionMatrix.rows-3; y++) { //Segmentation fault if i use y=0 (it is not possible to analyze with values ​​greater than 5 (image limits))
        for(int x=3; x<portionMatrix.cols-3; x++) {	
		if (pixelScan(portionMatrix,x,y)) {
			if(noiseEliminator(portionMatrix,x,y)) {
				keyPoints.push_back(Point(x,y+value));
			}
            	}
        }
    }

    return keyPoints;
}





void cornerDetection::showKeyPoints(Mat& image, vector<Point> points) { //Function used to draw the key points detected by the algorithm  

    int sz_keyPoints = (int)points.size();
    int i;
   

    for(i = 0; i<sz_keyPoints; i++) {
        circle(image, points[i], 1, Scalar(0,0,255), 1, 8, 0); //Circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
        							  //Present in OpenCV library
    }
   
   
}






















