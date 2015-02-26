// C++
#include <iostream>
#include <algorithm>
#include <numeric>	//accumulate
#include <string>
#include <fstream>
// OPENCV
#include <opencv2/opencv.hpp>

#include <windows.h>



/*---------------------------------------------------------------------------------------------------------------------
	La funzione drawHist restituisce un'immagine in bianco e nero con rappresentato l'istogramma passato come parametro.
	E' tutt'altro che perfetta, va resa molto pi� parametrica per ottenere un'immagine pi� customizzata sulle nostre
	esigenze, ma intanto dovrebbe dare un'idea di come sono fatti gli istogrammi.
---------------------------------------------------------------------------------------------------------------------*/
cv::Mat1b drawHist(const std::vector<double> &hist, cv::Size imageSize) {
	unsigned dimX = imageSize.width + 10;
	unsigned dimY = imageSize.height;
	unsigned binW = dimX/hist.size();
	cv::Mat1b histImg(dimY,dimX,uchar(0));
	cv::normalize(hist,hist,1.0,0.0,cv::NORM_MINMAX);
	for (unsigned i=0; i<hist.size(); ++i) {
		cv::Point p1(5+(i*binW),dimY-5);
		cv::Point p2(5+(i*binW)+binW,dimY-5 - unsigned(hist[i]*(dimY-10)));
		rectangle(histImg,p1,p2,cv::Scalar(255),-1);
	}
	return histImg;
}



/*---------------------------------------------------------------------------------------------------------------------
	La funzione quantize trasforma un istogramma da un numero di bin pari a old_dim
	ad un numero di bin pari a new_dim. Non sono sicuro al 100% che sia priva di bug. 
---------------------------------------------------------------------------------------------------------------------*/
void quantize(std::vector<double> &hist, double old_dim, double new_dim) {
	int scalef = floor(old_dim / new_dim);
	
	int k=1;
	for(int i=0;i<new_dim;++i) {
		for(int j=0;j<scalef-1;++j) {
			hist[i] += hist[k++]; 
		}
	}
	hist.resize(new_dim);
}



/*---------------------------------------------------------------------------------------------------------------------
	La funzione sumToOne normalizza un istogramma (vector<double>) facendo s� che la somma di tutti i valori
	contenuti nell'istogramma sia uguale a 1. Questa sono sicuro al 100% che sia priva di bug.
---------------------------------------------------------------------------------------------------------------------*/
void sumToOne (std::vector<double>& histo) {
	double total = std::accumulate(histo.begin(), histo.end(), 0.0);
	std::for_each(histo.begin(), histo.end(), [&total] (double &val) {if(total!=0) val /= total;});
}


/*---------------------------------------------------------------------------------------------------------------------
	La funzione roundToTen arrotonda un numero alla decina successiva o precedente a seconda che l'unit� sia maggiore
	o meno di cinque. Esempi: 143-> 140, 79->80 ecc.
	Serve a creare un istogramma di dimensione multipla di 10, in modo che la riduzione in 10 bin lavori bene.
---------------------------------------------------------------------------------------------------------------------*/
int roundToTen (int numberToRound) {
	int r = numberToRound % 10;
	if ( r < 5)
		return (numberToRound - r);
	else
		return (numberToRound + (10 - r));
}


/*---------------------------------------------------------------------------------------------------------------------
	La funzione computeFeatureVector serve a calcolare il vettore di feature di un frame.

	INPUT:	Mat &frame:						il frame di cui calcolare il feature vector
			Rect peopleRect:				il rettangolo in cui il PD ha individuato la persona
			int bins:						numero di bin del feature vector (pare vezzani usasse 10 bin)
			vector<double> featureVector:	il feature vector da riempire
			vector<Mat> &histogramImages:	vettore di due elementi che � riempito con le immagini dei due istogrammi
			bool createHistImages:			flag che indica se devono o meno essere create le immagini dei due istogrammi

	OUTPUT: void							(� tutto passato per reference, quindi sono modificati i parametri)

---------------------------------------------------------------------------------------------------------------------*/
void computeFeatureVector ( cv::Mat &frame, cv::Rect peopleRect, int bins, std::vector<double> &featureVector,
						   std::vector<cv::Mat> &histogramImages, bool createHistImages) {

	//	Per calcolare PI mi muovo da peopleRect.y a (peopleRect.y+peopleRect.height)
	//	scorrendo la silhouette per fette orizzontali.
	//	Per calcolare THETA mi muovo invece da peopleRect.x a (peopleRect.x+peopleRect.width)
	//	scorrendo la silhouette per fette verticali.
	
	int hist_pi_size = roundToTen(peopleRect.height);
	int hist_theta_size = roundToTen(peopleRect.width);
	std::vector<double> hist_pi		= std::vector<double>(hist_pi_size);
	std::vector<double> hist_theta	= std::vector<double>(hist_theta_size);

	int yOffset = peopleRect.y > 0 ? peopleRect.y : 0;
	int xOffset = peopleRect.x > 0 ? peopleRect.x : 0;

	for (int i = yOffset; i<(yOffset+hist_pi_size); ++i)
		hist_pi[i-yOffset] = countNonZero(frame.row(i));
	
	for (int i = xOffset; i<(xOffset+hist_theta_size); ++i) 
		hist_theta[i-xOffset] = countNonZero(frame.col(i));

	if(createHistImages){
		// Salva in un vettore le visualizzazioni grafiche dei due istogrammi
		// [Le istruzioni di transpose e flip ruotano il primo istogramma di 90� in senso orario]
		histogramImages[0] = drawHist(hist_pi, cv::Size(640,480));
		histogramImages[1] = drawHist(hist_theta, cv::Size(640,480));
		/*transpose(histogramImages[0], histogramImages[0]);  
		flip(histogramImages[0], histogramImages[0], 1); */
	}

	// Normalizzazione dei due istogrammi in modo che la somma dei valori sia = 1
	sumToOne(hist_pi);
	sumToOne(hist_theta);

	// Quantizzazione dei due istogrammi per ridurre il numero di bin a K/2
	quantize(hist_pi,	 hist_pi.size(),	bins/2);
	quantize(hist_theta, hist_theta.size(), bins/2);

	// Crea il feature vector (� passato per reference) concatenando i due istogrammi 'pi' e 'theta'
	featureVector = hist_pi;	
	featureVector.insert( featureVector.end(), hist_theta.begin(), hist_theta.end() );
}

/*---------------------------------------------------------------------------------------------------------------------
	La funzione computeFeatureVector serve a calcolare il vettore di feature di un frame.
	IN QUESTO CASO SI CONSIDERA L'INTERO FRAME, NON SOLO IL RETTANGOLO INDIVIDUATO DAL PEOPLE DETECTOR

	INPUT:	Mat &frame:						il frame di cui calcolare il feature vector
			Rect peopleRect:				il rettangolo in cui il PD ha individuato la persona
			int bins:						numero di bin del feature vector (pare vezzani usasse 10 bin)
			vector<double> featureVector:	il feature vector da riempire
			vector<Mat> &histogramImages:	vettore di due elementi che � riempito con le immagini dei due istogrammi
			bool createHistImages:			flag che indica se devono o meno essere create le immagini dei due istogrammi

	OUTPUT: void							(� tutto passato per reference, quindi sono modificati i parametri)

---------------------------------------------------------------------------------------------------------------------*/
void computeFeatureVector2 ( cv::Mat &frame, int bins, std::vector<double> &featureVector,
						   std::vector<cv::Mat> &histogramImages, bool createHistImages) {

	//	Per calcolare PI mi muovo da peopleRect.y a (peopleRect.y+peopleRect.height)
	//	scorrendo la silhouette per fette orizzontali.
	//	Per calcolare THETA mi muovo invece da peopleRect.x a (peopleRect.x+peopleRect.width)
	//	scorrendo la silhouette per fette verticali.
	
	int hist_pi_size = frame.rows; //prima era 480
	int hist_theta_size = frame.cols; //prima era 640
	std::vector<double> hist_pi		= std::vector<double>(hist_pi_size);
	std::vector<double> hist_theta	= std::vector<double>(hist_theta_size);

	for (int i = 0; i<hist_pi_size; ++i)
		hist_pi[i] = countNonZero(frame.row(i));
	
	for (int i = 0; i<hist_theta_size; ++i) 
		hist_theta[i] = countNonZero(frame.col(i));

	if(createHistImages){
		// Salva in un vettore le visualizzazioni grafiche dei due istogrammi
		// [Le istruzioni di transpose e flip ruotano il primo istogramma di 90� in senso orario]
		histogramImages[0] = drawHist(hist_pi, cv::Size(frame.cols,frame.rows));
		histogramImages[1] = drawHist(hist_theta, cv::Size(frame.cols,frame.rows));
		/*transpose(histogramImages[0], histogramImages[0]);  
		flip(histogramImages[0], histogramImages[0], 1); */
	}

	// Normalizzazione dei due istogrammi in modo che la somma dei valori sia = 1
	sumToOne(hist_pi);
	sumToOne(hist_theta);

	// Quantizzazione dei due istogrammi per ridurre il numero di bin a K/2
	quantize(hist_pi,	 hist_pi.size(),	bins/2);
	quantize(hist_theta, hist_theta.size(), bins/2);

	// Crea il feature vector (� passato per reference) concatenando i due istogrammi 'pi' e 'theta'
	featureVector = hist_pi;	
	featureVector.insert( featureVector.end(), hist_theta.begin(), hist_theta.end() );
}

void writeFeatureVectorToFile (std::string category, std::string outFileName, std::vector<double> featureVector)
{
	// Memento: se la directory � gi� esistente, CreateDirectory fallisce silenziosamente
	bool flag = CreateDirectory("training", NULL);
	std::string folder = "training\\" + category;
	std::string outPath = folder + "\\" + outFileName;
	flag = CreateDirectory(folder.c_str(), NULL);
	std::ofstream out(outPath, std::ios::out | std::ios::app);
	std::for_each(featureVector.begin(), featureVector.end(), [&out] (double val) {out << val << std::endl;});
	out.close();
}










