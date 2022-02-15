#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>	

#include <cornerDetection.h>	//Header file of the cornerDetection implementation contains class definitions
				//and functions, instead the implementation of the class goes into the .cpp file		
#include <iostream>

#include <chrono>	
	
	
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

	if(argc!=2) {
		cerr<<"ERROR: correct syntax is ./cornerDetection imageName.jpg"<<endl;		
		return -1;
	}

	cornerDetection corner;		

	Mat rootImage, keyPointImage;			/*Mat is a structure that maintains matrix / image characteristics (number of rows and columns, data type, etc.) and a pointer to the data.
					  		So nothing prevents us from having several instances of Mat corresponding to the same data. A Mat maintains a reference count indicating
					  		whether data should be deallocated when a particular Mat instance is destroyed.
							*/

	rootImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	keyPointImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	
	if(!rootImage.data) {
		cerr<<"ERROR: data have not been loaded correctly, could not open or find image"<<endl;
		return -1;
	}



	cout<<"findKeyPoints() invoked\n";
	corner.findKeyPoints(keyPointImage);
	
	cout<<"showKeyPoint() invoked\n";
	corner.showKeyPoints(rootImage);

	
	

	namedWindow("Displayed Image", CV_WINDOW_NORMAL); 	//This OpenCV function with CV_WINDOW_NORMAL allows to resize the window
	
	imshow("Displayed Image", rootImage);			//This OpenCV function allows to display the final result in a previously defined window
	
	

	waitKey(0);
	
	
	return 0;
}

	
