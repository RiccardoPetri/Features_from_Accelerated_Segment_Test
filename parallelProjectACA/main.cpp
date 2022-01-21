#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>	

#include <cornerDetection.h>	//Header file of the cornerDetection implementation contains class definitions
				//and functions, instead the implementation of the class goes into the .cpp file		
#include <iostream>
	
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {


	MPI_Status status;
	int myrank, size;
	int numbCols, cc,rest, restOverlap ;
	int rows, columns, sizeMaster;
	int sizeMasterTail = 0;
	vector<Point> keyPointsFound;
	vector<Point> keyMaster;
	vector<Point> keyMasterTail;
	vector<Point> keyFinal;
	Mat greyScaleImage; 
	Mat filledMatrix;
	Mat restMatrix;


	cornerDetection corner;	
	Mat rootImage, keyPointImage;		/*Mat is a structure that maintains matrix / image characteristics (number of rows and columns, data type, etc.) and a pointer to the data.
						  So nothing prevents us from having several instances of Mat corresponding to the same data. A Mat maintains a reference count indicating
						  whether data should be deallocated when a particular Mat instance is destroyed.
						*/

	// 1-Initialize MPI
	MPI_Init(&argc, &argv);
	// 2-Get the rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// 3-Get the total number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	MPI_Datatype pointStruct;
  	MPI_Type_contiguous(2, MPI_INT, &pointStruct);
  	MPI_Type_commit(&pointStruct);

	MPI_Datatype twoPointStruct;
  	MPI_Type_contiguous(1, pointStruct, &twoPointStruct);
  	MPI_Type_commit(&twoPointStruct);


	
	if (myrank == 0) {

		if(argc!=2) {
			cerr<<"ERROR: correct syntax is ./cornerDetection imageName.jpg"<<endl;		
			return -1;
		}
			
		rootImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);       //with these, we split the three rgb value of a pixel into a matrix, 	
		keyPointImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);   //so if we scan the new columns we have triplet of the r-g-b value of each pixel
		
		if(!rootImage.data) {
			cerr<<"ERROR: data have not been loaded correctly, could not open or find image"<<endl;
			return -1;
		}
	
		//cout<<"getGreyScaleImage() invoked\n";


		greyScaleImage = corner.getGreyScaleImage(keyPointImage); //this matrix must be splitted for all the processors


		columns = greyScaleImage.cols;
		rows = greyScaleImage.rows;

		//q = columns / size;	//number of column that will be sent
		cc = rows / size;
		rest = rows % size;
		restOverlap = 0;
		if(rest > 3) {
			restOverlap = rest+6;
		}
		printf("rest vale %d\n", rest);					//OKAY MA SE IL RESTO È 3 COME ORA? TROPPO POCO!!!
										//MENTRE SE NON FACCIO REST? NON SAREBBE MEGLIO?
		
		MPI_Bcast(&cc, 1, MPI_INT, 0, MPI_COMM_WORLD);
	


		//int vec[greyScaleImage.rows*greyScaleImage.cols] = { 0 };
		Mat flat = greyScaleImage.reshape(1, greyScaleImage.total());	//SE LO FACESSIMO APPENA CREATA LA MATRICE? (NEL METODO convertPixelToGreyScale())???
		vector<int> vec = greyScaleImage.isContinuous()? flat : flat.clone();

		//cout << greyScaleImage;

		int vecSize = static_cast<int>(vec.size());
		printf("aaa %d\n", vecSize);

		

		//int size_v = static_cast<int>(vec.size())/3;
	
		int v[vecSize];

	        
		for(int i=0; i<vecSize; i++) {		
			v[i] = (int)vec[i];	
		}					
		


		int new_cc = cc - 6;
		int new_ccc = cc + 6;

		
		// master sends only a portion of the original image to all slaves
		for (int p = 1; p < size; p++) {

			MPI_Send(&v[(((cc*p)-6)*columns)], (new_ccc)*columns, MPI_INT, p, 555, MPI_COMM_WORLD);						       
		}


		filledMatrix = Mat(cc, columns, CV_32S, &v[0]); //instead of the vec3b we have int (for each pixel)!!!
		

		if(rest > 3) {
			restMatrix = Mat(restOverlap, columns, CV_32S, &v[vecSize-(restOverlap*columns)]);
		}

		

		//master computes operation of the vector's head
		keyMaster = corner.pushKeyPointsInVector(filledMatrix, myrank, cc);
		sizeMaster = (int)keyMaster.size();

		
		
		if(rest > 3) {
			//master computes operation of the vectorìs tail
			keyMasterTail = corner.pushKeyPointsInVector(restMatrix, size, cc);
			sizeMasterTail = (int)keyMasterTail.size();
		}
	}



	if (myrank != 0) {

		
		MPI_Bcast(&numbCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
		//printf("I'm the slave %d; I receivedddd %d from process 0.\n", myrank, numbCols);


		MPI_Bcast(&cc, 1, MPI_INT, 0, MPI_COMM_WORLD);
		//printf("I'm the slave %d; I received %d from process 0.\n", myrank, cc);

		
		//int new_cc = cc - 6;
		int new_ccc = cc + 6;

		int numberOfCols = numbCols;

		int receiveVec[new_ccc*numberOfCols];
		//int receiveVec[cc*numberOfCols];
		
		Mat filledMatrix;

		// slaves receive a small vector...
		MPI_Recv(&receiveVec[0], (new_ccc)*numberOfCols, MPI_INT, 0, 555, MPI_COMM_WORLD, &status);
		//MPI_Recv(receiveVec, (cc)*numberOfCols, MPI_INT, 0, 555, MPI_COMM_WORLD, &status);

		filledMatrix = Mat(new_ccc, numberOfCols, CV_32S, &receiveVec[0]);
		//filledMatrix = Mat(new_ccc, numberOfCols, CV_32S, &receiveVec[0]);

		//cout << filledMatrix;

		keyPointsFound = corner.pushKeyPointsInVector(filledMatrix, myrank, cc);

		int sizeOfVec = (int)keyPointsFound.size();
		//printf("slalve %d keeeeeeeeeeeeeeyyyyyyyyyyyyy %d\n",myrank ,sizeOfVec);

		
		MPI_Send(&sizeOfVec, 1, MPI_INT, 0, 554, MPI_COMM_WORLD);
		

		MPI_Send(&keyPointsFound[0] , sizeOfVec, twoPointStruct, 0, 556, MPI_COMM_WORLD);
		
		



		//MPI_Barrier(MPI_COMM_WORLD);

	}



	if (myrank==0){	
		
		
			int sizeOfVec[size-1];
			int finalArrayDim = sizeMaster;
			
	
			for(int p=1; p<size; p++){

				MPI_Recv(&sizeOfVec[p-1], 1, MPI_INT, p, 554, MPI_COMM_WORLD, &status);
				finalArrayDim += sizeOfVec[p-1];
				
			}

			vector<Point> keyFinal(finalArrayDim+sizeMasterTail);	
			
			
			copy(keyMaster.begin(), keyMaster.end(), keyFinal.begin());
			

			if(restOverlap > 3) {
				copy(keyMasterTail.begin(), keyMasterTail.end(), &keyFinal[finalArrayDim]);
			}


			int dimension = sizeMaster;


			for(int p=1; p<size; p++){

				//printf("dimmmmm %d\n", dimension);
				//printf("zzzz %d\n", (int)sizeOfVec[p-1]);
			
				MPI_Recv(&keyFinal[dimension], (int)sizeOfVec[p-1], twoPointStruct, p, 556, MPI_COMM_WORLD, &status);
				//printf("DIM1 %d\n", dimension);
				dimension+= (int)sizeOfVec[p-1];
				//printf("DIM2 %d\n", dimension);
				//printf("rexxxxxxx %d\n", (int)sizeOfVec[p-1]);
			}

			//printf("ssssssssssssssssss %d\n", sizeMaster);

//__________________________________________________________________________________________________________________________________

/*
			for (int p = 1; p < size; p++){		
			printf("ssssssssssssssssss %d\n", (int)sizeOfVec[p-1]);
				for(int j=sizeMaster; j < sizeMaster+(int)sizeOfVec[p-1]; j++)	{
					


						int vall= (int)keyFinal[j].y;
						printf("vallll %d\n", vall);

						keyFinal[j].y = vall - 6;
						printf("vallll %d\n", vall);
				}
			sizeMaster += (int)sizeOfVec[p-1];
			}

/*
			for(int j = finalArrayDim; j < (finalArrayDim+sizeMasterTail) ; j++)	{
					
				keyFinal[j].y = keyFinal[j].y + (((size+1)*cc)-6);
			}
*/
//__________________________________________________________________________________________________________________________________


		//MPI_Barrier(MPI_COMM_WORLD);
		printf("ssssssssssssssssss %d\n", keyFinal.size());
		cout<<"showKeyPoint() invoked\n";
		corner.showKeyPoints(rootImage, keyFinal);	

		namedWindow("Displayed Image", CV_WINDOW_NORMAL); 	//This OpenCV function with CV_WINDOW_NORMAL allows to resize the window
		
		imshow("Displayed Image", rootImage);			//This OpenCV function allows to display the final result in a previously defined window
	

		waitKey(0);
	}

	MPI_Finalize();	//@@@@@ VEDERE SE METTERE ALLA FINE DELLE OPERAZ PARALLELE

	return 0;
}

	
	        
