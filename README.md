# deep-learning-challenge

##Overview of the Analysis

The purpose of this analysis is to build a binary classifier using machine learning that helps Alphabet Soup, a nonprofit foundation, decide which applicants are most likely to succeed if they receive funding. Using neural networks, we analyze data from over 34,000 organizations to identify high-potential applicants, ensuring funds go to those with the best chance of success.

##Results

##Data Preprocessing

* What variable(s) are the target(s) for your model?

The target variable for the model is the "IS_SUCCESSFUL" column.

* What variable(s) are the features for your model?

The features for the model include the following columns:

APPLICATION_TYPE: Alphabet Soup application type;
CLASSIFICATION: Government organization classification;
USE_CASE: Use case for funding;
ORGANIZATION: Organization type;
INCOME_AMT: Income classification;
ASK_AMT: Funding amount requested.

* What variable(s) should be removed from the input data because they are neither targets nor features?

EIN, NAME, SPECIAL_CONSIDERATIONS, AFFILIATION, and STATUS should be removed from the input data because they are neither targets nor features.

##Compiling, Training, and Evaluating the Model

* How many neurons, layers, and activation functions did you select for your neural network model, and why?

In first model I used two hidden layers because it allow efficient learning without overcomplicating the model Since an ideal starting point for neural networks is 2 to 4 hidden layers.Hidden Layer 1: 80 neurons, Hidden Layer 2: 30 neurons 
In second model I used three hidden layers with higher neurons allows the model to capture more complex patterns before refining predictions.Hidden Layer 1: 100 neurons, Hidden Layer 2: 80 neurons, Hidden Layer 3: 30 neurons.
In third model I used three hidden layers with less neurons allows the model to refine patterns efficiently.Hidden Layer 1: 84 neurons,Hidden Layer 2: 32 neurons, Hidden Layer 3: 16 neurons.
I used Activation Function ReLU for all three models because helps the network learn complex patterns while avoiding the vanishing gradient problem also is the most commonly used activation for hidden layers in deep learning.

* Were you able to achieve the target model performance?
 No, the target accuracy of 75% was not achieved while the models performed well, the highest accuracy reached was 72.5%

* What steps did you take in your attempts to increase model performance?
To increase model performance I tested multiple optimization techniques by tested three different architectures, adjusting the number of layers and neuron, dropped unnecessary columns (EIN, NAME, SPECIAL_CONSIDERATIONS, AFFILIATION, STATUS). I also experimented with different epoch values to find the optimal training duration for the model. However, when I increased the complexity by adding additional layers and setting a higher epoch count, the model crashed due to memory overload.This was likely caused by the large number of parameters and the higher computational demands, leading to excessive memory usage in Google Colab.to address this, I had to reduce the number of epochs and simplify the architecture to avoid memory issues while still achieving reasonable performance.

##Summary

 The deep learning model was designed to predict whether an applicant would be successful if funded by Alphabet Soup. Several architectures were tested, adjusting hidden layers, neurons and epochs to optimize performance while avoiding overfitting and memory issues.Best Accuracy Achieved was 72.5%.The results did not meet the 75% accuracy goal, so we need to try different model settings (hyperparameters) or test other models to build a more reliable prediction system.
 
 I recommend trying a Random Forest classifier for this classification problem. This model is well-suited for structured data, requires less computational power, and can handle large datasets efficiently. It may also achieve higher accuracy without the memory issues faced by the deep learning model.

##Additional Note

I want to mention that I used ChatGPT for assistance while working on Model 3. During this model I encountered a RAM crash when running the train-test split, caused by memory overload. With the help of ChatGPT, I was able to fix the issue by reducing memory usage. Instead of using .values, I used .to_numpy(dtype="float32"), which lowered memory consumption and prevented crashes.
y = numeric_df['IS_SUCCESSFUL'].to_numpy(dtype="float32")
X = numeric_df.drop(columns=['IS_SUCCESSFUL']).to_numpy(dtype="float32")














