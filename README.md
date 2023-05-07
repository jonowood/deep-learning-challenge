<p align="center">
<img src="/Images/deep-learning.jpeg" align="Center">
</p>

# Module 21 Challenge Submission - Deep Learning

#### This repository contains the code and resources for the Alphabet Soup Charity deep learning challenge. The goal of this project is to create a binary classifier to predict whether an applicant will be successful if funded by the charity using a deep learning neural network model.


## Table of Contents
- [Getting Started](#getting-started)
- [Project Deliverables](#project-deliverables)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Getting Started

Getting Started
To get started with this project, you will need to have Python 3 and several libraries installed, including Pandas, Matplotlib, Scikit-learn, TensorFlow, and Keras. You can install these libraries using pip or conda. Once you have the necessary libraries, you can retrieve the charity_data.csv dataset, which is located in the Resources folder. The main code for the project is in the AlphabetSoupCharity.ipynb and AlphabetSoupCharity_Optimisation.ipynb Jupyter Notebooks, which contain the steps for data preprocessing, model creation, evaluation, and optimization. You can run the code cells in the notebooks to replicate the analysis and generate the results. We recommend using an Anaconda environment to manage the dependencies and ensure reproducibility.

## Project Deliverables

#### Part 1: Data Preprocessing
The first part of the analysis involves data preprocessing, where we read the charity_data.csv dataset, identify the target and feature variables, drop the EIN and NAME columns, encode categorical variables, split the data into training and testing datasets, and scale the data using the StandardScaler function.

#### Part 2: Compile, Train, and Evaluate the Model
The second part of the analysis involves creating a deep learning neural network model with a defined number of input features and nodes for each layer. We add hidden layers and an output layer with appropriate activation functions, compile and train the model, and evaluate the model using the test data to determine the loss and accuracy. We then export the results to an HDF5 file named AlphabetSoupCharity.h5.

#### Part 3: Optimise the Model
The third part of the analysis involves optimizing the model by implementing various techniques such as changing the number of neurons, layers, optimizers, activation functions, and performing a more extensive hyperparameter search using grid search and Bayesian optimization. We save and export the optimized model results to an HDF5 file named AlphabetSoupCharity_Optimisation.h5.

<img src="/Images/Barplot.png">

<img src="/Images/Heatmap.png">

## Conclusion
The optimized deep learning neural network model achieves an accuracy of approximately 73%, which is slightly below the desired threshold of 75%. Despite trying different techniques to optimize the model, we were unable to achieve the desired accuracy. This suggests that there may be limitations in the dataset or that a different type of model may be better suited for this problem.

## Project Structure

```
AlphabetSoupCharity_Optimisation.ipynb
AlphabetSoupCharity_Preprocess.ipynb
Analysis Outputs
   |-- model_optimization_results.csv
Images
   |-- Barplot.png
   |-- Heatmap.png
   |-- deep-learning.jpeg
ML Models
   |-- AlphabetSoupCharity_Optimisation.h5
   |-- AlphabetSoupCharity_Preprocess.h5
README.md
Report
   |-- Alphabet Soup Charity Deep Learning Model Analysis.docx
   |-- Report - Alphabet Soup Charity Deep Learning Model Analysis.pdf
Requirements.txt
Resources
   |-- charity_data.csv
```
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments
I would like to thank our bootcamp instructors for their guidance and support throughout this assignment.

## References
- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
- Matplotlib: https://matplotlib.org/
- scikit-learn: https://scikit-learn.org/stable/
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- Seaborn: https://seaborn.pydata.org/
- Jupyter Notebook: https://jupyter.org/
- Anaconda: https://www.anaconda.com/
-	University of Western Australia Data Bootcamp: https://bootcamp.ce.uwa.edu.au/data/
