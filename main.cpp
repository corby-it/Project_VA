//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>
//C
#include <stdio.h>
#include <stdlib.h>
//C++
#include <iostream>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <cstdint>

using namespace cv;
using namespace std;

// ------------------ COSTANTI --------------------------------
const Size STD_SIZE(640, 480); // Risoluzione a cui viene resizato qualsiasi frame
const float MOG_LEARNING_RATE = 0.05f; // Learning rate della BG subtraction (sia per MOG che MOG2)

// ------------------ VARIABILI -------------------------------
Mat frame; //current frame
Mat frameDrawn; //frame su cui disegnare i rettangoli
Mat fgMaskMOG; //fg mask generated by MOG method
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
Ptr<BackgroundSubtractor> pMOG; //MOG Background subtractor
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
int keyboard;
int countTotalFrame = 0;
float avgBsTime = 0;
float avgPdTime = 0;
float fps = 0;

// Variabili utilizzate nell'estrazione dell'area di interesse
int leftX = 0, rightX = 0;

// Inizializzazione utile nel caso non trovi contorni
Mat3b frameResized = Mat3b(STD_SIZE.height, 250);

// Dichiarazione delle funzioni
void help();
void processVideo(char* videoFilename);


template <typename T>  bool IsInBounds(const T& value, const T& low, const T& high) {
	return !(value < low) && !(high < value);
}


// ------------------ MAIN -------------------------------
int main(int argc, char* argv[])
{
	//print help information
	help();

	////check for the input parameter correctness
	//if(argc != 3) {
	//    cerr <<"Incorret input list" << endl;
	//    cerr <<"exiting..." << endl;
	//    return EXIT_FAILURE;
	//}

	//create GUI windows
	namedWindow("Frame");
	namedWindow("FG Mask MOG");
	//namedWindow("FG Mask MOG 2");
	namedWindow("Background Subtraction and People Detector");

	//create Background Subtractor objects
	pMOG= new BackgroundSubtractorMOG(); //MOG approach
	pMOG2 = new BackgroundSubtractorMOG2(); //MOG2 approach

	if(strcmp(argv[1], "-vid") == 0) {
		//input data coming from a video
		double t = (double)getTickCount();
		processVideo(argv[2]);
		t = (double)getTickCount() - t; 
		fps += t*1000./cv::getTickFrequency();
	}
	else {
		//error in reading input parameters
		cerr <<"Please, check the input parameters." << endl;
		cerr <<"Exiting..." << endl;
		return EXIT_FAILURE;
	}

	//Tempi utili per prestazioni (attuali: BS=10.7, PD=36.3, FPS=12.7)
	cout << "Tempo medio per la Background Subtraction: " << avgBsTime/countTotalFrame << endl;
	cout << "Tempo medio per la People Detection: " << avgPdTime/countTotalFrame << endl;
	cout << "FPS: " << countTotalFrame/(fps/1000) << endl;

	// pause, wait for key
	system("pause");
	
	//destroy GUI windows
	destroyAllWindows();

	return EXIT_SUCCESS;
}

// ------------------ VIDEO PROCESSING -------------------------------
void processVideo(char* videoFilename) {

	// Hog detector
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//create the capture object
	VideoCapture capture(videoFilename); // o 0 per webcam!
	if(!capture.isOpened()){
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}

	//read input data. ESC or 'q' for quitting
	while( (char)keyboard != 'q' && (char)keyboard != 27 ) {
		//read the current frame
		if(!capture.read(frame)) {
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			return; //Altrimenti esce di botto
			//exit(EXIT_FAILURE);
		}

		//Conteggio numero frame totali
		countTotalFrame++;

		// Resize dei frame in input alla dimensione standard
		resize(frame, frame, STD_SIZE);

		//Per la mia webcam messa male
		//flip(frame, frame, -1);

		//Copio il frame per ottenere quello su cui disegnare i rettangoli
		frameDrawn = frame.clone();

		// BACKGROUND SUBTRACTION --------------------------------------------

		double s = (double)getTickCount();
		pMOG->operator()(frame, fgMaskMOG, MOG_LEARNING_RATE);
		//pMOG2->operator()(frame, fgMaskMOG2, MOG_LEARNING_RATE);
		s = (double)getTickCount() - s;
		avgBsTime += s*1000./cv::getTickFrequency();


		// FILTERING
		medianBlur(fgMaskMOG, fgMaskMOG, 15);
		dilate(fgMaskMOG, fgMaskMOG, Mat(), Point(-1, -1), 2, 1, 1);

		// disegna una bounding box BLU attorno alle zone di foreground
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		findContours( fgMaskMOG, contours, hierarchy, RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS);

		// ---------------------------------------------------------------------------------------------
		// Trova il centro di massa di ogni contorno (trovato dopo il filtering)
		// Calcola poi il centroide della nuvola di punti per stabilire il punto centrale del movimento
		// Se tolti i commenti, in giallo i centri di massa dei contorni. In rosso il centroide complessivo.
		int centroidX, centroidY;
		if(contours.size() > 0) {
			vector<int> cmContoursX, cmContoursY;
			for ( size_t i=0; i<contours.size(); ++i ){
				Moments mo = moments(contours[i], true);
				Point2d result = Point2d(mo.m10/mo.m00 , mo.m01/mo.m00);
				// Controlla che il centro di massa del contorno sia nel range dell'immagine
				if( IsInBounds(int(result.x), 0, STD_SIZE.width) && IsInBounds(int(result.y), 0, STD_SIZE.height)){
					cmContoursX.push_back(result.x);
					cmContoursY.push_back(result.y);
					// [DEBUG] Disegna la posizione del centro di massa e del boundingRect del contorno
					rectangle(frameDrawn, boundingRect(contours[i]), Scalar(255,0,0), 1);
					circle(frameDrawn, result, 3, Scalar(0,255,255), 3);
				}
			}

			// Trova il contorno di area maggiore per poter dare un maggior peso alla sua posizione
			int largestContourIndex = -1;
			int largestArea = -1;
			for (int i=0; i<contours.size(); ++i) {
				if(contourArea(contours[i]) > largestArea) {
					largestArea = contourArea(contours[i]);
					largestContourIndex = i;
				}
			}
			// [DEBUG] Disegna il boundingRect del contorno di area maggiore
			rectangle(frameDrawn, boundingRect(contours[largestContourIndex]), Scalar(255,0,0), 3);


			// Calcola la posizione del centroide
			// Il centro del contorno di area maggiore pesa N volte pi� degli altri nella media
			// (Per capire il codice vedere il terzo parametro della funzione di accumulate)
			uint32_t largestContourWeight = 3;
			centroidX = accumulate(cmContoursX.begin(), cmContoursX.end(), (largestContourWeight-1)*cmContoursX[largestContourIndex])
				/ (contours.size()+(largestContourWeight-1));
			centroidY = accumulate(cmContoursY.begin(), cmContoursY.end(), (largestContourWeight-1)*cmContoursY[largestContourIndex])
				/ (contours.size()+(largestContourWeight-1));
			// [DEBUG] Visualizza il centroide (in rosso)
			circle(frameDrawn, Point2d(centroidX, centroidY), 7, Scalar(0,0,255), 3);

			// Dimensioni della ROI (questa parte si pu� collassare nella successiva secondo me)
			leftX = centroidX - 125;
			if(leftX < 0)
				leftX = 0;
			rightX = centroidX + 125;
			if(rightX > STD_SIZE.width)
				rightX = STD_SIZE.width;

			// Se il centroide � all'interno del frame, ritaglia la ROI
			if (IsInBounds(centroidX, 0, STD_SIZE.width) && IsInBounds(centroidY, 0, STD_SIZE.height)) {
				Rect newRect = Rect(leftX, 0, abs(rightX-leftX), STD_SIZE.height);
				frameResized = Mat3b(STD_SIZE.height, 250);
				frameResized = frame(newRect);
			}
		}
		// ---------------------------------------------------------------------------------------------


		// HOG PEOPLE DETECTION -----------------------------------------------
		vector<Rect> found, found_filtered;
		double t = (double)getTickCount();
		// run the detector with default parameters. to get a higher hit-rate
		// (and more false alarms, respectively), decrease the hitThreshold and
		// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
		hog.detectMultiScale(frameResized, found, 0, Size(8,8), Size(0,0), 1.05, 2);
		t = (double)getTickCount() - t; 
		//cout << "detection time = " << t*1000./cv::getTickFrequency() << " - found objects: " << found.size() << endl;
		avgPdTime += t*1000./cv::getTickFrequency();

		size_t i, j;
		for( i = 0; i < found.size(); i++ ) {
			Rect r = found[i];
			for( j = 0; j < found.size(); j++ )
				if( j != i && (r & found[j]) == r)
					break;
			if( j == found.size() )
				found_filtered.push_back(r);
		}

		// se trova pi� di una persona scorre tutti i risulati e tiene il rettangolo il cui centro
		// � pi� vicino al centroide di movimento (quello che pi� probabilmente contiene la persona reale),
		// in questo modo si elimina la possibilit� di avere due persone detected in scena.
		if(found_filtered.size() > 0){
			Point2d movementCentroid(centroidX, centroidY);

			Rect closestRect = found_filtered[0];
			Point2d closestRectCenter(closestRect.x + closestRect.width/2 , closestRect.y + closestRect.height/2 );
			double closestDistance = norm(closestRectCenter - movementCentroid);

			for(i = 1; i < found_filtered.size(); i++){
				Rect currRect = found_filtered[i];
				Point2d currRectCenter(currRect.x + currRect.width/2 , currRect.y + currRect.height/2 );
				double currDistance = norm(currRectCenter - movementCentroid);

				if(currDistance < closestDistance){
					closestDistance = currDistance;
					closestRect = currRect;
				}
			}

			// the HOG detector returns slightly larger rectangles than the real objects.
			// so we slightly shrink the rectangles to get a nicer output.
			closestRect.x += cvRound(closestRect.width*0.1) + leftX;
			closestRect.width = cvRound(closestRect.width*0.8);
			closestRect.y += cvRound(closestRect.height*0.07);
			closestRect.height = cvRound(closestRect.height*0.8);
			rectangle(frameDrawn, closestRect.tl(), closestRect.br(), cv::Scalar(0,255,0), 4);

		}

		// Disegna tutti i risultati della people detection

		//for( i = 0; i < found_filtered.size(); i++ ){
		//	Rect r = found_filtered[i];
		//	// the HOG detector returns slightly larger rectangles than the real objects.
		//	// so we slightly shrink the rectangles to get a nicer output.
		//	r.x += cvRound(r.width*0.1) + leftX;
		//	r.width = cvRound(r.width*0.8);
		//	r.y += cvRound(r.height*0.07);
		//	r.height = cvRound(r.height*0.8);
		//	rectangle(frameDrawn, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
		//}

		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("frameResized", frameResized);
		imshow("FG Mask MOG", fgMaskMOG);
		//imshow("FG Mask MOG 2", fgMaskMOG2);
		imshow("Background Subtraction and People Detector", frameDrawn);

		//get the input from the keyboard
		keyboard = waitKey(1);

	} // end of while

	//delete capture object
	capture.release();


}




void help()
{
	cout
		<< "--------------------------------------------------------------------------"  << endl
		<< "Usage:"                                                                      << endl
		<< "./bs {-vid <video filename>|-img <image filename>}"                          << endl
		<< "for example: ./bs -vid video.avi"                                            << endl
		<< "or: ./bs -img /data/images/1.png"                                            << endl
		<< "--------------------------------------------------------------------------"  << endl
		<< endl;
}