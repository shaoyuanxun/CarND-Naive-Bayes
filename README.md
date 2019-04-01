# CarND-Naive-Bayes

## 1. Naive Bayes for Car Behavior Predictions in C++
In this project, I implement a Gaussian Naive Bayes classifier to predict the behavior of vehicles on a highway. In the image below you can see the behaviors you'll be looking for on a 3 lane highway (with lanes of 4 meter width). The dots represent the d (y axis) and s (x axis) coordinates of vehicles as they either.

![alt text](image/img1.png)

* change lanes left (shown in blue)
* keep lane (shown in black)
* change lanes right (shown in red)

The Naive Bayes Classifier predicts which of these three maneuvers a vehicle is engaged in given a single coordinate (sampled from the trajectories shown below). Four features are given: s, d, s_dot, d_dot.

##  2. Gaussian Naive Bayes
* **Compute the conditional probabilities for each feature/label combination**. For a feature x and label C with mean μ and standard deviation σ ​​​:
<p align="center">
  <img src = image/dist.svg>
</p>
&nbsp; &nbsp; &nbsp; &nbsp; Here v is the value of feature x in the new data point.

* **Use the conditional probabilities in a Naive Bayes Classifier**.
<p align="center">
<img src = image/argmax.svg>
</p>
&nbsp; &nbsp; &nbsp; &nbsp; In this formula, the argmax is taken over all possible labels C​k​​​  and the product is taken over all features xi with values vi.

## 3. Compiling and Running

The main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./NBC
