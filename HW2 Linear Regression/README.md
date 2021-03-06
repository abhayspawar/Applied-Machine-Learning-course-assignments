# homework-ii-abhayspawar


Abhay Pawar
UNI: asp2197

Final R squared on test data = 0.571

I followed following steps to build the model:

1. Since there are a lot a categorical variables, dummies need to be created. I checked if the categorical features contain all the categories in both train and test. Since, the data is large enough this is true. Hence, there is no need to create a pipeline. I created dummies on the whole dataset. Implementing a pipeline will not change the results for this dataset.
I dropped all the continuous variables, variables related to householder and created dummies for the categorical features on the data. I built a LASSO on these dummy variables and only kept those variables with non-zero coefficients

2. Then, I tried adding the contiuous features and re-ran the model to see if there is any improvement. I only kept those continuous variables which added value. I used Ridge and LASSO for this pupose. Most of the continuous variables didn't add much value and finally, I added only 3 of these variables. I tried various imputation strategies wherever the data wasn't reported. But, it did not add much value. The 3 variables that I kept were number of rooms, number of bedrooms and total yearly cost. For total yearly cost, I imputed 9999 with 0. This should be okay, because there is another variable which has a different category for all 9999 valued houses and this variable will give different prediction for 9999 valued houses than 0 valued houses. I wasn't sure if total yearly cost variable is related to the householder. I still kept it in the model. Removing this variable would drop the R squared by around 0.001 (1%).

For LASSO and ridge, I used grid search to find the optimal parameters. Some of the variables like number of units could be used as it is because they are ordinal. But, converting them into dummies gave better results. Finally, the model that I am using to predict is Ridge.