#pragma once

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
#include <vector>
#include <utility>
#include <string>
//
//#include "FrameAnalyzer.h"
#include "dirent.h"
#include "utils.h"
#include "config.h"
#include "gmmstd_hmm_GMM.h"
#include "gmmstd_gmm_tiny.h"


class HMMTester {
public:
	std::vector<gmmstd::CHMM_GMM> vHMM;
	std::vector<std::vector<double>> vFeatures;
	std::vector<std::string> classAction;
	int best;
	double loglk;

	HMMTester(std::vector<gmmstd::CHMM_GMM> vHMM, std::vector<std::string> classAction){
		this->vHMM = vHMM;
		this->classAction = classAction;
	}

	std::size_t HMMTester::countFrame () {
		return vFeatures.size();
	}

	std::pair<double, std::string> HMMTester::getClassification (){
		std::string c = classAction[best].substr(classAction[best].find_first_of("_")+1, classAction[best].length()-classAction[best].find_first_of("_"));
		return std::pair<double,std::string>(loglk, c); 
	}

	void HMMTester::testingHMM(std::vector<double> featureVector){

		//-----------------TESTING----------------------
		//Fatto solo se c'è una bounding box valida (DA OTTIMIZZARE)

		//Per ottenere likelihood massima
		double max = DBL_MIN;
		double max_other = DBL_MIN;
		best = 0;

		//Accumulo le features frame per frame
		vFeatures.push_back(featureVector);

		//Per ogni HMM trovato nella cartella
		for(int i=0;i<vHMM.size();++i){

			//Ottengo matrice di transizione
			cv::Mat_<double> A;
			A = vHMM[i].m_A; 

			//Valuto l'HMM
			typedef std::vector<std::vector<double>>::iterator iter_vf;
			const iter_vf ivf_init = vFeatures.begin();
			const iter_vf ivf_final = vFeatures.end();
			//Calcolo la logLikelihood
			//cout << "Calcolo la logLikelihood dell'HMM: " << tmp  << " basandomi su: " << vFeatures.size() << " frame" << endl;
			//cout << "Calcolo lk con frame " << vFeatures.size() << endl;
			loglk = vHMM[i].LogLikelihood(ivf_init, ivf_final, &A);
			//cout << tmp << "\t" << loglk << endl;

			////Genero alcune stringhe utili
			//string file_name = filename;
			//file_name = file_name.substr(file_name.find_last_of("\\")+1, file_name.length()-file_name.find_last_of("\\")-5);

			//Verifico se è massimo e se l'hmm non è lo stesso dell'azione
			if((loglk>max && loglk==loglk)/* && (classAction[i].compare(file_name) != 0)*/){
				max = loglk;
				best = i;
			}

			//string tipo = classAction[best].substr(classAction[best].find_first_of("_")+1, classAction[best].length()-classAction[best].find_first_of("_"));
			//string tipo_file = file_name.substr(file_name.find_first_of("_")+1, file_name.length()-file_name.find_first_of("_"));

			////Trovo il valore massimo di un hmm di un altro tipo di azione
			//if((loglk>max_other && loglk==loglk) && (tipo.compare(tipo_file) != 0))
			//	max_other = loglk;

		}//fine while

		//cout << "CLASSIFICAZIONE: " << classAction[best].substr(classAction[best].find_first_of("_")+1, classAction[best].length()-classAction[best].find_first_of("_"));
		/*cout << " SICURO AL: " << (max/max_other)*100 << "%" << endl;*/

		//system("pause");
	}

};