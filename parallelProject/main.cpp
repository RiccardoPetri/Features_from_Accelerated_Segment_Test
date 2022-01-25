#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>
#include <chrono>	

#include <cornerDetection.h>	//Header file of the cornerDetection implementation contains class definitions
				//and functions, instead the implementation of the class goes into the .cpp file		
#include <iostream>
	
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

	auto start = chrono::steady_clock::now();
    	/* Do your stuff here */
	MPI_Status status;
	int myrank, size;
	int numbCols, PortionRows,rest, restOverlap ;
	int rows, columns, sizeMaster;
	int sizeMasterTail = 0;
	vector<Point> keyPointsFound;		//Vector used to store keypoints found by slaves
	vector<Point> keyMaster;		//Vector used to store keypoints by Master
	vector<Point> keyMasterTail;		//Vector used to store keypoints of the rest of the matrix by Master
	vector<Point> keyFinal;			//Vector used to store every keypoint at the end
	Mat greyScaleImage; 			//Reference to the converted image in greyscale
	Mat filledMatrix;			//Portion of image analyzed by slaves
	Mat restMatrix;				//Portion of matrix analyzed by the master due to the rest of division


	cornerDetection corner;	
	Mat rootImage, keyPointImage;	

	// 1-Initialize MPI
	MPI_Init(&argc, &argv);
	// 2-Get the rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// 3-Get the total number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	MPI_Datatype pointStruct;
  	MPI_Type_contiguous(2, MPI_INT, &pointStruct);		//Struct used to send and receive keypoints
  	MPI_Type_commit(&pointStruct);


	
	if (myrank == 0) {

		

		if(argc!=2) {
			cerr<<"ERROR: correct syntax is ./cornerDetection imageName.jpg"<<endl;		
			return -1;
		}
			
		rootImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);       //With these, we split the three rgb value of a pixel into a matrix, 	
		keyPointImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);   //so if we scan the new columns we have triplet of the r-g-b value of each pixel
		
		if(!rootImage.data) {
			cerr<<"ERROR: data have not been loaded correctly, could not open or find image"<<endl;
			return -1;
		}
	


		greyScaleImage = corner.getGreyScaleImage(keyPointImage); //This matrix will be splitted among all processors


		columns = greyScaleImage.cols;
		rows = greyScaleImage.rows;
	
		PortionRows = rows / size; //Number of rows that will be sent
		rest = rows % size; //Portion of rows remaining in Master after the splitting
		restOverlap = 0; 
		if(rest > 3) {
			restOverlap = rest+6;
		}				
		
		MPI_Bcast(&PortionRows, 1, MPI_INT, 0, MPI_COMM_WORLD);		//Broadcasting the number of rows 
	


		Mat flat = greyScaleImage.reshape(1, greyScaleImage.total());		//Converting the matrix into a vector to send pixel values
		vector<int> vec = greyScaleImage.isContinuous()? flat : flat.clone();

		int vecSize = static_cast<int>(vec.size());

	
		int v[vecSize];

	        
		for(int i=0; i<vecSize; i++) {		
			v[i] = (int)vec[i];	
		}					
		

		int PortionRows_Overlap = PortionRows + 6;

		
		// Master sends only a portion of the original image to all the slaves
		for (int p = 1; p < size; p++) {

			MPI_Send(&v[(((PortionRows*p)-6)*columns)], (PortionRows_Overlap)*columns, MPI_INT, p, 555, MPI_COMM_WORLD);						       
		}


		filledMatrix = Mat(PortionRows, columns, CV_32S, &v[0]); //Matrix that will be analyzed by Master
		

		if(rest > 3) {
			restMatrix = Mat(restOverlap, columns, CV_32S, &v[vecSize-(restOverlap*columns)]);	//Portion of matrix remaining from the splitting will be analyzed by Master
		}

		

		//Master computes operation of the vector's head
		keyMaster = corner.pushKeyPointsInVector(filledMatrix, myrank, PortionRows);
		sizeMaster = (int)keyMaster.size();

		
		
		if(rest > 3) {
			//Master computes operation of the vector's tail
			keyMasterTail = corner.pushKeyPointsInVector(restMatrix, size, PortionRows);
			sizeMasterTail = (int)keyMasterTail.size();
		}
	}



	if (myrank != 0) {

		
		MPI_Bcast(&numbCols, 1, MPI_INT, 0, MPI_COMM_WORLD);


		MPI_Bcast(&PortionRows, 1, MPI_INT, 0, MPI_COMM_WORLD);

		
		int PortionRows_Overlap = PortionRows + 6;

		int numberOfCols = numbCols;

		int receiveVec[PortionRows_Overlap*numberOfCols];  //Vector used to store pixel values from the Master's send
		
		Mat filledMatrix;

		MPI_Recv(&receiveVec[0], (PortionRows_Overlap)*numberOfCols, MPI_INT, 0, 555, MPI_COMM_WORLD, &status);


		filledMatrix = Mat(PortionRows_Overlap, numberOfCols, CV_32S, &receiveVec[0]);

		keyPointsFound = corner.pushKeyPointsInVector(filledMatrix, myrank, PortionRows);	//This method stores found keypoints into a vector

		int sizeOfVec = (int)keyPointsFound.size();

		
		MPI_Send(&sizeOfVec, 1, MPI_INT, 0, 554, MPI_COMM_WORLD);				//Send vector dimension to Master
		

		MPI_Send(&keyPointsFound[0] , sizeOfVec, pointStruct, 0, 556, MPI_COMM_WORLD);		//Used to send back results to Master
		

	}



	if (myrank==0){	
		
		
			int sizeOfVec[size-1];
			int finalArrayDim = sizeMaster;
			
	
			for(int p=1; p<size; p++){

				MPI_Recv(&sizeOfVec[p-1], 1, MPI_INT, p, 554, MPI_COMM_WORLD, &status);		//Master receives sizes of each vector from slaves
				finalArrayDim += sizeOfVec[p-1];
				
			}

			vector<Point> keyFinal(finalArrayDim+sizeMasterTail);	
			
			
			copy(keyMaster.begin(), keyMaster.end(), keyFinal.begin());			//Insert all keypoints in a unique vector
			

			if(restOverlap > 3) {
				copy(keyMasterTail.begin(), keyMasterTail.end(), &keyFinal[finalArrayDim]);
			}


			int dimension = sizeMaster;


			for(int p=1; p<size; p++){
			
				MPI_Recv(&keyFinal[dimension], (int)sizeOfVec[p-1], pointStruct, p, 556, MPI_COMM_WORLD, &status);	//Master receives keypoints from slaves
				dimension+= (int)sizeOfVec[p-1];
			}


		//MPI_Barrier(MPI_COMM_WORLD);
		cout<<"showKeyPoint() invoked\n";
		corner.showKeyPoints(rootImage, keyFinal);	

		auto end = chrono::steady_clock::now();
		auto diff = end - start;
		cout << chrono::duration <double, milli> (diff).count() << " ms" << endl;


		namedWindow("Displayed Image", CV_WINDOW_NORMAL); 	//This OpenCV function with CV_WINDOW_NORMAL allows to resize the window
		
		imshow("Displayed Image", rootImage);			//This OpenCV function allows to display the final result in a previously defined window
	
		
		waitKey(0);
	}
	
	MPI_Finalize();
	


	return 0;
}

	
	        
