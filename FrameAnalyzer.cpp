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

#include "FrameAnalyzer.h"

using namespace std;
using namespace cv;

FrameAnalyzer::FrameAnalyzer(char* videoFilename, int mog)
	: MOG_LEARNING_RATE(0.05f), STD_SIZE(Size(640,480)), RED(Scalar(0,0,255)), GREEN(Scalar(0,255,0)), BLUE(Scalar(255,0,0)),
	filename(videoFilename), mogType(mog) {

	// inizializzazione variabili
	predictionVect = Point2d(0, 0);

	leftX = 0;
	rightX = 0;
	xOffset = 0;

	avgBsTime = 0;
	avgPdTime = 0;

	// Inizializzazione utile nel caso non trovi contorni
	frameResized = Mat3b(STD_SIZE.height, 250);

	// crea le finestre dell'interfaccia
	namedWindow("Frame");
	namedWindow("FG Mask MOG");
	namedWindow("Background Subtraction and People Detector");

	// impostazione del background suppressor
	switch (mogType){
	case 0:
		pMOG = new BackgroundSubtractorMOG(); break; //MOG approach
	case 1:
		pMOG = new BackgroundSubtractorMOG2(); break; //MOG2 approach
	}

	// imposto il pepole detector
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	// crea l'oggetto capture
	capture = VideoCapture(filename); // o 0 per webcam!
	if(!capture.isOpened()){
		// errore nell'aprire il file in input
		cerr << "Impossibile aprire il file video: " << filename << endl;
		// TODO magari si potrebbe fare qualcosa di più user-friendly piuttosto che chiudere tutto il programma...
		exit(EXIT_FAILURE);
	}

}

int FrameAnalyzer::getFrameCount(){
	if(capture.isOpened()){
		return capture.get(CV_CAP_PROP_FRAME_COUNT);
	}
	else return -1;
}

int FrameAnalyzer::getCurrentFramePos(){
	if(capture.isOpened()){
		return capture.get(CV_CAP_PROP_POS_FRAMES);
	}
	else return -1;
}

const VideoCapture& FrameAnalyzer::getCapture(){
	return capture;
}

void FrameAnalyzer::release(){
	//delete capture object
	capture.release();
}

// Ritorna true se è andato tutto bene, false se non è riuscita a leggere un frame (cioè il video è finito)
bool FrameAnalyzer::processFrame() {

	//read the current frame
	if(!capture.read(frame)) {
		cerr << "Video terminato." << endl;
		return false; //Altrimenti esce di botto
	}

	// Resize dei frame in input alla dimensione standard
	resize(frame, frame, STD_SIZE);

	//Per la webcam messa male di mak
	//flip(frame, frame, -1);

	//Copio il frame per ottenere quello su cui disegnare i rettangoli
	frameDrawn = frame.clone();

	// BACKGROUND SUBTRACTION --------------------------------------------

	double s = (double)getTickCount();
	pMOG->operator()(frame, fgMaskMOG, MOG_LEARNING_RATE);
	s = (double)getTickCount() - s;
	avgBsTime += s*1000./cv::getTickFrequency();


	// FILTERING
	medianBlur(fgMaskMOG, fgMaskMOG, 15);
	dilate(fgMaskMOG, fgMaskMOG, Mat(), Point(-1, -1), 2, 1, 1);

	// disegna una bounding box BLU attorno alle zone di foreground
	std::vector<std::vector<cv::Point> > contours;
	std::vector<std::vector<cv::Point> > inBoundContours;
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
				inBoundContours.push_back(contours[i]);
				// [DEBUG] Disegna la posizione del centro di massa e del boundingRect del contorno
				rectangle(frameDrawn, boundingRect(contours[i]), Scalar(255,0,0), 1);
				circle(frameDrawn, result, 3, Scalar(0,255,255), 3);
			}
		}


		// Trova il contorno di area maggiore per poter dare un maggior peso alla sua posizione
		int largestContourIndex = -1;
		int largestArea = -1;
		for (int i=0; i<inBoundContours.size(); ++i) {
			if(contourArea(inBoundContours[i]) > largestArea) {
				largestArea = contourArea(inBoundContours[i]);
				largestContourIndex = i;
			}
		}
		// [DEBUG] Disegna il boundingRect del contorno di area maggiore
		rectangle(frameDrawn, boundingRect(inBoundContours[largestContourIndex]), BLUE, 3);


		// Calcola la posizione del centroide
		// Il centro del contorno di area maggiore pesa N volte più degli altri nella media
		// (Per capire il codice vedere il terzo parametro della funzione di accumulate)
		uint32_t largestContourWeight = 3;
		centroidX = accumulate(cmContoursX.begin(), cmContoursX.end(), (largestContourWeight-1)*cmContoursX[largestContourIndex])
			/ (contours.size()+(largestContourWeight-1));
		centroidY = accumulate(cmContoursY.begin(), cmContoursY.end(), (largestContourWeight-1)*cmContoursY[largestContourIndex])
			/ (contours.size()+(largestContourWeight-1));
		// [DEBUG] Visualizza il centroide (in rosso)
		circle(frameDrawn, Point2d(centroidX, centroidY), 7, RED, 3);

		// Dimensioni della ROI (questa parte si può collassare nella successiva secondo me)
		leftX = centroidX - 125;
		if(leftX < 0)
			leftX = 0;
		rightX = centroidX + 125;
		if(rightX > STD_SIZE.width)
			rightX = STD_SIZE.width;

		// Se il centroide è all'interno del frame, ritaglia la ROI
		if (IsInBounds(centroidX, 0, STD_SIZE.width) && IsInBounds(centroidY, 0, STD_SIZE.height)) {
			Rect newRect = Rect(leftX, 0, abs(rightX-leftX), STD_SIZE.height);
			frameResized = Mat3b(STD_SIZE.height, 250);
			frameResized = frame(newRect);
		}
	}
	// ---------------------------------------------------------------------------------------------


	// HOG PEOPLE DETECTION ------------------------------------------------------------------------
	// people detection solo sui frame pari
	if( ((int)capture.get(CV_CAP_PROP_POS_FRAMES)) % 4 == 0){

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

		// se trova più di una persona scorre tutti i risulati e tiene il rettangolo il cui centro
		// è più vicino al centroide di movimento (quello che più probabilmente contiene la persona reale),
		// in questo modo si elimina la possibilità di avere due persone detected in scena.
		if(found_filtered.size() > 0){

			xOffset = leftX;

			Point2d movementCentroid(centroidX, centroidY);

			tmpClosestRect = found_filtered[0];
			Point2d closestRectCenter(tmpClosestRect.x + tmpClosestRect.width/2 , tmpClosestRect.y + tmpClosestRect.height/2 );
			double closestDistance = norm(closestRectCenter - movementCentroid);

			for(i = 1; i < found_filtered.size(); i++){
				Rect currRect = found_filtered[i];
				Point2d currRectCenter(currRect.x + currRect.width/2 , currRect.y + currRect.height/2 );
				double currDistance = norm(currRectCenter - movementCentroid);

				if(currDistance < closestDistance){
					closestDistance = currDistance;
					tmpClosestRect = currRect;
				}
			}

			// se ho trovato il rettangolo più vicino aggiorno il predictionVect

			predictionVect.x = sign(tmpClosestRect.x - closestRect.x);
			predictionVect.y = sign(tmpClosestRect.y - closestRect.y);

			closestRect = tmpClosestRect;

			// Disegna il rettangolo sul frame
			drawRectOnFrameDrawn(closestRect, frameDrawn, GREEN, 4, xOffset);

		}
		else {
			// frame pari ma non è stata trovata la persona
			if((predictionVect.x != 0) || (predictionVect.y!=0)) {
				closestRect.x += predictionVect.x;
				closestRect.y += predictionVect.y;
				drawRectOnFrameDrawn(closestRect, frameDrawn, GREEN, 4, xOffset);

			}
		}
	}
	else {
		// frame dispari
		if((predictionVect.x != 0) || (predictionVect.y!=0)) {
			closestRect.x += predictionVect.x;
			closestRect.y += predictionVect.y;
			// Disegna il rettangolo sul frame
			drawRectOnFrameDrawn(closestRect, frameDrawn, GREEN, 4, xOffset);
		}
	}


	//show the current frame and the fg masks
	imshow("Frame", frame);
	imshow("frameResized", frameResized);
	imshow("FG Mask MOG", fgMaskMOG);
	//imshow("FG Mask MOG 2", fgMaskMOG2);
	imshow("Background Subtraction and People Detector", frameDrawn);

	//get the input from the keyboard
	keyboard = waitKey(1);
	
	return true;
}

void FrameAnalyzer::drawRectOnFrameDrawn( Rect closestRect, Mat frameDrawn, cv::Scalar color, int thickness, int xOffset) {

	closestRect.x += cvRound(closestRect.width*0.1) + xOffset;
	closestRect.width = cvRound(closestRect.width*0.8);
	closestRect.y += cvRound(closestRect.height*0.07);
	closestRect.height = cvRound(closestRect.height*0.8);
	rectangle(frameDrawn, closestRect.tl(), closestRect.br(), color, thickness);

}

FrameAnalyzer::~FrameAnalyzer(void){}