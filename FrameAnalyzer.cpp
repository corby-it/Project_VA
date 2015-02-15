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

#include "utils.h"

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

	// inizializzo il background
	bgName = getBgName(filename);

	VideoCapture bgCapture(bgName);
	if(!bgCapture.isOpened()){
		// errore nell'aprire il file di background
		cerr << "Impossibile aprire il file di background: " << bgName << endl;
		// TODO magari si potrebbe fare qualcosa di pi� user-friendly piuttosto che chiudere tutto il programma...
		exit(EXIT_FAILURE);
	}
	else {
		// leggo il primo frame dal file di background e lo metto in frameBg (resizato)
		bgCapture.read(frameBg);
		resize(frameBg, frameBg, STD_SIZE);
	}

	// crea l'oggetto capture
	capture = VideoCapture(filename); // o 0 per webcam!
	if(!capture.isOpened()){
		// errore nell'aprire il file in input
		cerr << "Impossibile aprire il file video: " << filename << endl;
		// TODO magari si potrebbe fare qualcosa di pi� user-friendly piuttosto che chiudere tutto il programma...
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

// Ritorna true se � andato tutto bene, false se non � riuscita a leggere un frame (cio� il video � finito)
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

	// vecchia bg subtraction fatta con MOG
	// pMOG->operator()(frame, fgMaskMOG, MOG_LEARNING_RATE);

	Mat tmpDiff;
	Mat1b tmpDiffGray;

	// sottraggo il BG al frame corrente
	absdiff(frame, frameBg, tmpDiff);
	//imshow("tmpDiff", tmpDiff);

	// converto l'immagine differenza in scala di grigi
	cvtColor(tmpDiff, tmpDiffGray, CV_RGB2GRAY);
	//imshow("tmpDiffGray", tmpDiffGray);

	// soglia di otsu su tmpDiffGray
	threshold(tmpDiffGray, fgMaskMOG, 128, 255, CV_THRESH_OTSU);

	s = (double)getTickCount() - s;
	avgBsTime += s*1000./cv::getTickFrequency();

	// FILTERING e MORFOLOGIA SU fgMaskMOG per ottenere una silhouette migliore 
	medianBlur(fgMaskMOG, fgMaskMOG, 15);
	/*dilate(fgMaskMOG, fgMaskMOG, Mat(), Point(-1, -1), 2, 1, 1);*/

	// disegna una bounding box BLU attorno alle zone di foreground
	std::vector<std::vector<cv::Point> > contours;
	std::vector<std::vector<cv::Point> > inBoundContours;
	std::vector<cv::Vec4i> hierarchy;
	// passo un clone di fgMaskMOG per fare in modo che non la modifichi
	findContours( fgMaskMOG.clone(), contours, hierarchy, RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS);

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
		// Il centro del contorno di area maggiore pesa N volte pi� degli altri nella media
		// (Per capire il codice vedere il terzo parametro della funzione di accumulate)
		uint32_t largestContourWeight = 3;
		centroidX = accumulate(cmContoursX.begin(), cmContoursX.end(), (largestContourWeight-1)*cmContoursX[largestContourIndex])
			/ (contours.size()+(largestContourWeight-1));
		centroidY = accumulate(cmContoursY.begin(), cmContoursY.end(), (largestContourWeight-1)*cmContoursY[largestContourIndex])
			/ (contours.size()+(largestContourWeight-1));
		// [DEBUG] Visualizza il centroide (in rosso)
		circle(frameDrawn, Point2d(centroidX, centroidY), 7, RED, 3);

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

		// se trova pi� di una persona scorre tutti i risulati e tiene il rettangolo il cui centro
		// � pi� vicino al centroide di movimento (quello che pi� probabilmente contiene la persona reale),
		// in questo modo si elimina la possibilit� di avere due persone detected in scena.
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

			// se ho trovato il rettangolo pi� vicino aggiorno il predictionVect

			predictionVect.x = sign(tmpClosestRect.x - closestRect.x);
			predictionVect.y = sign(tmpClosestRect.y - closestRect.y);

			closestRect = tmpClosestRect;

			// Disegna il rettangolo sul frame
			drawRectOnFrameDrawn(closestRect, frameDrawn, GREEN, 4, xOffset);

		}
		else {
			// frame pari ma non � stata trovata la persona
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


	// -------------------- CALCOLO DELL'ISTOGRAMMA--------------------------------
	int numberBins = 10;
	vector<double> featureVector(numberBins, 0);

	bool createThe2HistogramImages = false;
	vector<Mat> histogramImages(2);

	if(closestRect.area()>0){
		computeFeatureVector ( fgMaskMOG, closestRect, numberBins, featureVector, histogramImages, createThe2HistogramImages );
		if(createThe2HistogramImages) {
			for(int i=0; i<histogramImages.size(); ++i)
				imshow("Histogram "+to_string(i+1), histogramImages[i]);
		}
	}

	//show the current frame and the fg masks
	imshow("Frame", frame);
	imshow("frameResized", frameResized);
	imshow("FG Mask MOG - Silhouette", fgMaskMOG);
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

string FrameAnalyzer::getBgName(char* filename){

	stringstream ss;
	ss << "backgrounds/";

	string strFname(filename);
	strFname = strFname.substr(strFname.find_first_of("/")+1);

	if(strcmp(strFname.data(), "daria_bend.avi") == 0 || strcmp(strFname.data(), "daria_jack.avi") == 0 ||strcmp(strFname.data(), "daria_wave1.avi") == 0 ||strcmp(strFname.data(), "daria_wave2.avi") == 0){
		ss << "bg_015.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "denis_bend.avi") == 0 || strcmp(strFname.data(), "daria_jack.avi") == 0 || strcmp(strFname.data(), "denis_pjump.avi") == 0 || strcmp(strFname.data(), "denis_wave1.avi") == 0 || strcmp(strFname.data(), "denis_wave2.avi") == 0 ){
		ss << "bg_026.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "eli_bend.avi") == 0 || strcmp(strFname.data(), "eli_pjump.avi") == 0 ||strcmp(strFname.data(), "eli_wave1.avi") == 0 ||strcmp(strFname.data(), "eli_wave2.avi") == 0){
		ss << "bg_062.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "ido_bend.avi") == 0 || strcmp(strFname.data(), "ido_jack.avi") == 0 || strcmp(strFname.data(), "ido_pjump.avi") == 0 || strcmp(strFname.data(), "ido_wave1.avi") == 0 || strcmp(strFname.data(), "ido_wave2.avi") == 0){
		ss << "bg_062.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "ira_bend.avi") == 0 || strcmp(strFname.data(), "ira_jack.avi") == 0 || strcmp(strFname.data(), "ira_pjump.avi") == 0 || strcmp(strFname.data(), "ira_wave1.avi") == 0 || strcmp(strFname.data(), "ira_wave2.avi") == 0){
		ss << "bg_007.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "lena_bend.avi") == 0 || strcmp(strFname.data(), "lena_pjump.avi") == 0 || strcmp(strFname.data(), "lena_wave1.avi") == 0 || strcmp(strFname.data(), "lena_wave2.avi") == 0 || strcmp(strFname.data(), "daria_pjump.avi") == 0){
		ss << "bg_038.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "lyova_bend.avi") == 0 || strcmp(strFname.data(), "lyova_jack.avi") == 0 || strcmp(strFname.data(), "lyova_pjump.avi") == 0 || strcmp(strFname.data(), "lyova_wave1.avi") == 0 || strcmp(strFname.data(), "lyova_wave2.avi") == 0){
		ss << "bg_046.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "moshe_bend.avi") == 0 || strcmp(strFname.data(), "moshe_pjump.avi") == 0 ||strcmp(strFname.data(), "moshe_wave1.avi") == 0 ||strcmp(strFname.data(), "moshe_wave2.avi") == 0){
		ss << "bg_070.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "shahar_walk.avi") == 0 || strcmp(strFname.data(), "shahar_pjump.avi") == 0 ||strcmp(strFname.data(), "shahar_wave1.avi") == 0 ||strcmp(strFname.data(), "shahar_wave2.avi") == 0){
		ss << "bg_079.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "moshe_jack.avi") == 0 ){
		ss << "moshe_bg_run.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "shahar_jack.avi") == 0 || strcmp(strFname.data(), "eli_jack.avi") == 0){
		ss << "shahar_bg_run.avi";
		return ss.str();
	}
	else if(strcmp(strFname.data(), "lena_jack.avi") == 0 ){
		ss << "lena_bg_jack.avi";
		return ss.str();
	}
	else { // se non ne trova nessuno ritorna il video originale
		ss.clear();
		ss << "dataset/" << filename;
		return ss.str();
	}
}

FrameAnalyzer::~FrameAnalyzer(void){}