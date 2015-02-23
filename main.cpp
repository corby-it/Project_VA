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
#include <string>
#include <vector>
#include <fstream>
// FrameAnalyzer
#include "FrameAnalyzer.h"

using namespace cv;
using namespace std;

// Dichiarazione delle funzioni
void help();
void videoProcessing(char* filename, string category);
vector<string> parseDatasetFile(string datasetFileName);

// ------------------ MAIN -------------------------------
int main(int argc, char* argv[])
{
	//print help information
	help();

	//check for the input parameter correctness
	if(argc != 3) {
		cerr <<"Incorret input list" << endl;
		cerr <<"exiting..." << endl;
		system("pause");
		return EXIT_FAILURE;
	}

	bool processALL = true;
	if(processALL) {
		vector<string> datasetLines = parseDatasetFile("dataset.txt");
		for(size_t i=0; i<datasetLines.size(); ++i)
		{
			string category = datasetLines[i].substr(0, datasetLines[i].find("|"));
			string filePath = datasetLines[i].substr((datasetLines[i].find("|")+1), datasetLines[i].length());
			videoProcessing(&filePath[0u], category);
		}
	}


	if(strcmp(argv[1], "-vid") == 0) {
		// inizia il processing del video
		videoProcessing(argv[2], "NULL");

	}
	else {
		//error in reading input parameters
		cerr <<"Please, check the input parameters." << endl;
		cerr <<"Exiting..." << endl;
		system("pause");
		return EXIT_FAILURE;
	}

	system("pause");

	//destroy GUI windows
	destroyAllWindows();
	system("pause");
	return EXIT_SUCCESS;
}

void videoProcessing(char* filename, string category){

	float fps = 0;

	// inizializzo l'oggetto che analizzerà il video
	FrameAnalyzer frameAnalyzer(filename, category);

	// stampo info sul video
	cout << "Analisi Video:" << endl;
	cout << "Nome file: " << filename << endl;
	cout << "Frame count: " << frameAnalyzer.getFrameCount() << endl << endl;

	// scorro tutto il video finche non finisce o finchè non premo 'q' o 27
	double t = (double)getTickCount();
	while(((char)frameAnalyzer.keyboard != 'q' && (char)frameAnalyzer.keyboard != 27)){

		// quando arriva alla fine esco comunque dal while
		if(!frameAnalyzer.processFrame()) break;
		waitKey(1);
	}
	t = (double)getTickCount() - t; 
	fps += t*1000./cv::getTickFrequency();

	//Tempi utili per prestazioni (attuali: BS=10.7, PD=36.3, FPS=12.7)
	cout << "Tempo medio per la Background Subtraction: " << frameAnalyzer.avgBsTime/frameAnalyzer.getFrameCount() << endl;
	cout << "Tempo medio per la People Detection: " << frameAnalyzer.avgPdTime/frameAnalyzer.getFrameCount() << endl;
	cout << "FPS: " << frameAnalyzer.getFrameCount()/(fps/1000) << endl;

	// faccio il release del videoCapture per ultima cosa altrimenti le proprietà (tipo il frameCount) che vado a leggere sono tutte sbagliate.
	frameAnalyzer.release();

}


vector<string> parseDatasetFile(string datasetFileName)
{
	string line;
	vector<string> datasetLines;

	ifstream inFile(datasetFileName);

	while (getline(inFile, line))
		datasetLines.push_back(line);

	return datasetLines;
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