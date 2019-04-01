#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

const std::map<std::string, int> GNB::label_values= {{"left", 0}, {"keep", 1}, {"right", 2}};

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/

    int c_value;
    double s, d, s_dot, d_dot;
    // double means[n_features];
    int n_data = labels.size();

    // init
    for (int j = 0; j < N_CLASSES; j++) {
        class_count[j] = 0;
        for (int k = 0; k < N_FEATURES; k++) {
            means[j][k] = 0;
            vars[j][k] = 0;
        }
    }

    // Calculate the means
    for (int i = 0; i< n_data; i++) {
        c_value = label_values.at(labels[i]);
        std::cout << "training label " << labels[i] << "," << c_value
                  << "(" << data[i][0]<< ", "
                  << data[i][1] << ", "
                  << data[i][2] << ", "
                  << data[i][3] << ") "
                  << std::endl;
        class_count[c_value]++;
        for (int k =0; k < N_FEATURES; k++) {
            means[c_value][k]+=data[i][k];
        }
    }

    for (int j = 0; j < N_CLASSES; j++) {
        for (int k =0; k < N_FEATURES; k++) {
            means[j][k] /= class_count[j];
        }
    }

    // Calculate the variances
    double diff;
    for (int i = 0; i < n_data; i ++) {
        c_value = label_values.at(labels[i]);
        for (int k = 0; k < N_FEATURES; k++) {
            diff =  data[i][k] - means[c_value][k];
            vars[c_value][k] += diff * diff;
        }
    }

    for (int j = 0 ; j < N_CLASSES; j++) {
        for (int k = 0; k < N_FEATURES; k++) {
            vars[j][k] /= class_count[j];
        }
        priors[j] = float(class_count[j]) / n_data;
    }

    cout << "total data counts: " << n_data << std::endl;
    for (int j = 0; j < N_CLASSES; j++) {
        std::cout << "priors" << j << ":" << priors[j] << std::endl;
        std::cout << "class counts " << j << ":" << class_count[j] << std::endl;
        std::cout << "means " << std::endl;
        for (int k = 0; k < N_FEATURES; k++) {
            std::cout << k << ":" << means[j][k] << " ";
        }
        std::cout << std::endl;
        std::cout << "vars ";
        for (int k = 0; k < N_FEATURES; k++) {
            std::cout << k << ":" << vars[j][k] << " ";
        }
        std::cout << std::endl;
    }
}

string GNB::predict(vector<double> obs)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
	*/

    std::cout << "predictions: ";
    for (int j = 0; j < N_CLASSES; j++) {
        posteriors[j] = priors[j];
        for (int k = 0; k < N_FEATURES; k++) {
            posteriors[j] *= gauss(obs[k], means[j][k], vars[j][k]);
        }
        std::cout << priors[j] << " " << posteriors[j] << " ";
    }
    std::cout << std::endl;

    int max_index = std::distance(std::begin(posteriors),
            std::max_element(std::begin(posteriors),
                             std::end(posteriors)));

    cout << "predicts " << max_index << " " << possible_labels[max_index] << std::endl;

	return this->possible_labels[max_index];

}

double GNB::gauss(double x, double m, double v) {
    double d = x - m;
    return 1/sqrt(2*M_PI*v)*exp(-1.0 *(d*d)/(2*v));
}
