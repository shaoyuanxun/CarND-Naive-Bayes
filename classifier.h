#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

class GNB {
public:
  static const int N_FEATURES = 4;
	static const int N_CLASSES  = 3;
	vector<string> possible_labels = {"left","keep","right"};
  double means[N_CLASSES][N_FEATURES];
	double vars[N_CLASSES][N_FEATURES];
	double priors[N_CLASSES];
	double posteriors[N_CLASSES];
  int class_count[N_CLASSES];
  int n_data;

  static const std::map<std::string, int> label_values;

	/**
  	* Constructor
  	*/
	GNB();

	/**
 	* Destructor
 	*/
	virtual ~GNB();

	void train(vector<vector<double>> data, vector<string>  labels);

	string predict(vector<double>);

	double gauss(double x, double mu, double v);

};

#endif



