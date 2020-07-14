# Spark ML pipeline for Twitter scrapper
The aim of the project was to add to a distributed Twitter scrapper implemented with cellery, RabbitMQ and Twint a machine leraning pipeline. The application works in dockerized environment. 

## Subtasks

1. **MongoDB connection** 

2. **Regression pipeline**
    - select a tweet attribute which is dependent variable
    - map data to contain a dependent variable and features vector columns
    - create ML pipeline
    - evaluate regressor using RMSE on train and test sets

3. **Binary classification pipeline**
    - select a tweet attribute that is a class
    - map data to contain class and features vector columns
    - create ML pipeline
    - evaluate classifier computing F1 metric

4. **Multi-class classification pipeline**
    - select a submission attribute which is a multi-class
    - map data to contain class and features vector columns
    - create ML pipeline
    - evaluate classifier using MulticlassClassificationEvaluator