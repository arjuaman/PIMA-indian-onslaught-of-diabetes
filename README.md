## Flask Implementation: 
Install the requirements.txt file using : <br>
<em> pip install -r requirements.txt --no-index --find-links file:///tmp/packages </em><br>

Run the pred.py file using: <br>
<em> python pred.py </em>

Note: Remove the following part from pred.py if you don't want to run in <em>Debug Mode</em>:<br>
<em> if __name__ == '__main__': </em> <br>
<em>    app.run(debug=True) </em>

<strong> If you want to train on any other ML Model, simply train on the dataset and use it's pickle file in <em>pred.py</em> file. </strong>

## Description:
Notebook2 contains the final project.

I used the Pima Indians onset of diabetes dataset. 
This is a standard machine learning dataset from the UCI Machine Learning repository.
It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.

As such, it is a binary classification problem (onset of diabetes as 1 or not as 0). 
All of the input variables that describe each patient are numerical.

I've added the dataset csv file as well as data description file too.

Input variables I used and their respective column names are:

1. Number of times pregnant : times_pregnant
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test : plasma_glucose_conc
3. Diastolic blood pressure (mm Hg) : diastolic_bp
4. Triceps skin fold thickness (mm) : triceps_skin_fold_thickness
5. 2-Hour serum insulin (mu U/ml) : insulin
6. Body mass index (weight in kg/(height in m)^2) : bmi
7. Diabetes pedigree function : diabetes_pedigree_function
8. Age (years) : age
9. Class variable(0 or 1) : target

I used the read_csv method of pandas library to load the dataset, with sep=',' as the delimiter.

Then I normalized it using StandardScaler() preprocessing.

Afterward I split the data into X_train, X_test, y_train, y_test respectively, with shapes being,
Train set: (460, 8) (460,)
Test set: (307, 8) (307,)

Models in Keras are defined as a sequence of layers.
We create a Sequential model and add layers.
relu activation function is added to the hidden layers, and sigmoid to the last layer, i.e. output layer.

Then we compile the model.

We will use cross entropy as the loss argument. 
This loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“.

We will define the optimizer as the efficient stochastic gradient descent algorithm “adam“. 
This is a popular version of gradient descent because it automatically tunes itself and gives good results in a wide range of problems. 

Finally, because it is a classification problem, we will collect and report the classification accuracy, defined via the metrics argument.

I used 200 epochs with a batch_size of 10.

Then we train the model.

Now we have trained our neural network on the train set and we can evaluate the performance of the network on the test set.

The evaluate() function returns a list with two values.
The first will be the loss of the model on the dataset and the second will be the accuracy of the model on the dataset.

Our model has an accuracy of 90% on train set.

The jaccard similarity score of our model on test set is 0.79153 , which is pretty good considering the size of the dataset.

Also I've plotted the confusion matrix.

The log loss is 0.650223.

We are using a sigmoid activation function on the output layer, so the predictions will be a probability in the range between 0 and 1.
We call the predict_classes() function on the model to predict crisp classes directly.

Hence, we're able to successfully build and train a 4-layered keras model and successfully do the predictions and fulfill our task.
