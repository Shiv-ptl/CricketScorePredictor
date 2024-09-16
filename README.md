# Cricket_Score_Predictor
ABSTRACT

Cricket is a sport that has gained popularity worldwide, with T20 and ODI formats attracting millions of viewers. Cricket enthusiasts are always interested in predicting the outcome of matches. In this study, we propose a machine learning-based approach to predict the scores of cricket matches. We will gather and preprocess match data, including team and player statistics, match conditions, and weather. We will consider various machine learning models, such as Linear Regression, Decision Tree, XGBoost, etc., for prediction. We will also calculate cross-validation scores and sensitivity using different machine learning algorithms. Data analysis and visualization are crucial stages of predictive modeling, which we will perform before prediction. We will select the best-performing algorithm to develop a predictive model. The purpose of this study is to demonstrate the potential of machine learning in predicting the outcomes of complex sporting events like T20 and ODI.

Keywords:
    • Numpy
    • Panda
    • Seaborn
    • One Hot Encoding
    • Scatter plot
    • Histogram
    • MAE, MSE, RMSE
    
INRTODUCTION

The T20 and ODI cricket leagues are globally recognized for their excitement and competitiveness, drawing millions of viewers worldwide. Cricket's popularity has led fans to eagerly anticipate match outcomes, employing diverse strategies for predictions due to the game's unpredictability.

This project aims to utilize machine learning algorithms for predicting T20 and ODI match scores accurately. Considering factors like team and player statistics, match location, and weather conditions, the goal is to develop a model with high predictive capability.

Existing methods for score prediction often rely on simplistic statistical models and historical data analysis, neglecting crucial variables such as changes in team composition, player injuries, and match conditions.

The report is structured as follows: Firstly, an overview of cricket matches and the challenges in score prediction will be provided. Subsequently, the existing literature on cricket score prediction will be reviewed, emphasizing the gaps in current approaches and the potential of machine learning techniques. Following this, the methodology employed in this project, encompassing data collection, pre-processing, and model training, will be detailed. Finally, the results of experiments will be presented, along with discussions on their implications for cricket score prediction.

BASIC CONCEPTS

2.1) Linear Regression:
Linear regression that depicts the relationship between a dependent variable and one or more independent variables is used in the statistical technique of linear regression to model the relationship between two continuous variables. Finding the intercept and slope parameters that reduce the discrepancy between observed and anticipated values is the objective. It is commonly used in various fields to analyze and predict the relationship between variables.



2.2) Random Forest:
A machine learning approach called random forest is employed for feature selection, regression, and classification applications. It is an ensemble learning technique that integrates various decision trees to produce a model that is more reliable and accurate. In a random forest, a random portion of the training data is used to train each decision tree, and a random subset of the feature choices is made at each node. The projections of all the trees are combined to get the final prediction.



Ensemble uses two types of methods:
Bagging
It creates a different training subset from sample training data with replacement & the final output is based on majority voting. For example,  Random Forest.
Boosting
It combines weak learners into strong learners by creating sequential models such that the final model has the highest accuracy. For example,  ADA BOOST, XG BOOST.



2.3) XGBoost (Extreme Gradient Boosting):
XGBoost (short for Extreme Gradient Boosting) is a machine learning algorithm used for classification, regression, and ranking tasks. It is based on a method called gradient boosting, which combines a number of weak models into a single, more reliable ensemble model. XGBoost uses a tree-based approach and provides several optimization techniques to improve model performance and reduce over fitting 

BASIC CONCEPTS, PROBLEM STATEMENT & REQUIREMENT SPECIFICATION

Predicting the scores of T20 and ODI matches can be achieved using various machine learning techniques or algorithms, including Logistic Regression, Gaussian Naive Bayes, K Nearest Neighbours, Support Vector Machines (SVM), Gradient Boosting, Decision Trees, and Random Forests.

Project Planning
The project can be divided into the following seven primary steps: 
	-Recognize the dataset. 
	-Make the data clean. 
 	- Examine the potential columns for features. 
	-Handle the features in the way that the model or algorithm specifies. 
	-Use training data to train the model or algorithm. 
	 -Use test data to evaluate the model or algorithm. 
	-Optimize the model or algorithm for increased precision.

Objective of the Project
The main objective of this project i.e. T20 and ODI cricket score prediction, using machine learning, is to build a model that can nearly predict the score of an T20 and ODI match based on various factors such as bat_team, bowl_team, runs_last_5,wicket_last_5 etc. The model should be able to handle the dynamic nature of the sport and provide predictions in real-time during the match.

System Design
A design serves as a vital blueprint for future constructions, particularly in software development where it plays a pivotal role. It involves translating requirements into a software representation, known as software design, which is encouraged during the design phase of software engineering. The ongoing system design phase entails crafting the new system based on user requirements and a comprehensive analysis of the current system.

To accurately translate customer requirements into the final software solution, design is paramount. It entails developing a representation or model that provides insights into the system's architecture, interfaces, and necessary components for implementation. The logical system design from systems analysis is transformed into a physical system design.

Model Phases

  • Prerequisite Analysis
  • Framework Design
  • Coding
  • Usage
  • Testing
  • Support

Data Extraction

Here we are performing various data preprocessing and feature engineering tasks on cricket match data stored in pandas DataFrames. Here's a breakdown of what's happening:

1. Loading Data: Loading cricket match data from pickle files into pandas Data Frames.
2. Data Cleaning:
   - Handling missing values in the 'city' column by inferring them from the 'venue' column.
   - Converting 'player_dismissed' values to 0 where they are None.
3. Filtering Data:
   - Filtering out cities that have hosted fewer than 5 matches (600 balls).
  
 Feature Engineering:

   - Calculating 'current_score' as the cumulative sum of runs scored.
   - Extracting 'over' and 'ball_no' from the 'ball' column.
   - Calculating 'balls_bowled' from 'over' and 'ball_no'.
   - Calculating 'balls_left' as the difference between the total balls in an innings (120) and 'balls_bowled'.
   - Calculating 'wickets_left' as the remaining wickets in an innings.
   - Calculating 'crr' (current run rate) as (current_score * 6) / 'balls_bowled'.
   - Computing 'last_five' as the sum of runs scored in the last five overs using a rolling sum.

- Data Aggregation:
   	- Grouping the data by match_id and aggregating runs scored for each match.

-Final DataFrame Creation:
  - Merging aggregated runs data with the original DataFrame.
  - Selecting relevant columns for the final DataFrame.
  -Data Shuffling: Shuffling the final DataFrame to remove bias.
  -Saving Data: Saving the final DataFrame to a new pickle file.

This process is repeated for two different datasets, likely representing different types of cricket matches (e.g., T20 matches and One Day Internationals). The resulting processed datasets are saved in separate pickle files ('dataset_level3.pkl' and 'dataset_level3_odi.pkl').

MODEL TRAINING:

Here we are performing machine learning tasks using regression models to predict cricket match scores based on various features.
Here we automates the process of data preprocessing, model building, evaluation, and visualization for predicting cricket match scores using regression models, providing insights into model performance and aiding in model selection.

1. Importing Libraries: Importing necessary libraries including pandas, pickle, numpy, seaborn, matplotlib.pyplot, and various modules from scikit-learn and XGBoost for machine learning tasks.

2. Loading Data: Loading preprocessed cricket match data from a pickle file into a pandas DataFrame.

3. Data Preparation:
    - Splitting the data into features (X) and target variable (y) for both regular matches and One Day Internationals (ODIs).
    - Further splitting the data into training and testing sets.

4. Feature Engineering: 
   - Utilizing ColumnTransformer to perform one-hot encoding on categorical variables ('batting_team', 'bowling_team', 'city').
   
5. Model Building:
   - Constructing pipelines for different regression models including XGBoost, Random Forest, and Linear Regression.
   - Each pipeline consists of data preprocessing steps (one-hot encoding and standardization) and the respective regression model.

6. Model Training and Evaluation:
   - Training each regression model pipeline on the training data.
   - Making predictions on the test data and evaluating model performance using R-squared score and mean absolute error.
   - Visualizing the distribution of residuals (the differences between predicted values and actual values) using seaborn's dis plot.

7. Saving Models: 
   - Saving trained models using pickle for potential future use.

8. Plotting Scatter Plots:
   - Plotting scatter plots to visualize the relationship between true scores and predicted scores for each regression model.

9. Repeat for ODIs:
   - The same steps are repeated for the dataset related to One Day Internationals (ODIs) with slight modifications in data handling and model building.

MODEL IMPLEMENTATION & WEB APP DEVELOPMENT
Using Streamlit in python to build a website that takes user input regarding current match, and predict and tell final score using our model with best R-squared score ie XGB.
Here's a breakdown of what inputs are collected and how they are used in the prediction process:

User Inputs:
Batting Team: Selection of the team batting.
Bowling Team: Selection of the team bowling.
City: Selection of the city where the match is taking place.
Current Score: Input of the current score of the batting team.
Overs Done: Number of overs completed in the match.
Wickets Out: Number of wickets fallen.
Runs scored in Last 5 Overs: Runs scored by the batting team in the last 5 overs.

Session State:
	The application uses session_state to persist the match type selection (T20s or ODIs). 	This allows users to switch between these match types without losing their other input 	selections.

Calculations:
Balls Left: Calculated based on the total overs (either 20 for T20s or 50 for ODIs) minus the overs done, considering the number of balls in an over (6 balls per over).
Wickets Left: Derived from subtracting the number of wickets out from the total of 10.
Current Run Rate (CRR): Calculated as the ratio of the current score to the overs completed (current_score / overs).

DataFrame Creation:
User inputs are organized into a pandas DataFrame (input_df) to match the format expected by the machine learning model.
This DataFrame includes:
    batting_team
    bowling_team
    city
    current_score
    balls_left
    wickets_left
    crr
    last_five

Prediction:
	Depending on the selected match type (T20s or ODIs), the application uses the 	appropriate pre-trained machine learning model (pipe_t20 or pipe_odi) to predict the 	score.
	The input DataFrame (input_df) is fed into the selected model (pipe_t20 or pipe_odi) 	to obtain the predicted score.

Displaying Results:
	The predicted score is then displayed using Streamlit's interface as a header, showing 	the estimated score based on the user's inputs and the trained machine learning model.


Here we have predicted the score of India after 38th over of India vs New Zealand ODI W.C. semi-final held in Mumbai.
And according to our model Predicted Score Came out to be :
380 Runs

And according to ESPNcricinfo website the predicted score after 38th over was 375 Runs

Where actually India scored 397 runs, so we can say that our model was more close to the actual score than ESPN’s.


CONCLUSION & FUTURE SCOPE

Conclusion

Selecting the optimal cricket team significantly impacts its success. This study aims to analyze T20 and ODI cricket data to predict team performance. Three classification algorithms were compared for accuracy using Jupyter and Anaconda navigator. XGBoost emerged as the most dependable classifier, achieving an accuracy rate of 94.3% for T20 and 95.6% for ODI matches.

