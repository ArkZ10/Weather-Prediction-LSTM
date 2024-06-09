# Weather Prediction using Time Series Model

This project focuses on predicting maximum temperatures in London using a time series model. The dataset used contains historical weather data, specifically maximum temperatures recorded over a period.

## Project Overview

The objective is to build a machine learning model that forecasts future maximum temperatures based on past data. The model architecture involves using convolutional and recurrent neural network layers to capture temporal patterns in the data.

## Dataset

The dataset (`london_weather.csv`) includes columns for date and maximum temperature. Initial preprocessing involves loading the data, handling missing values, and preparing it for modeling.

## Plotting the Time Series

Before modeling, the dataset is visualized using matplotlib to understand the trends and patterns in maximum temperature over time.

## Splitting the Dataset

The dataset is split into training and validation sets. The training set is used to train the model, while the validation set evaluates its performance on unseen data.

## Modeling

The model architecture includes:
- Convolutional layer to extract features from the sequence.
- LSTM (Long Short-Term Memory) layer for capturing long-term dependencies.
- Dense layers for further processing and prediction.

Training involves setting up a windowed dataset to feed into the model, optimizing hyperparameters, and monitoring metrics such as Mean Absolute Error (MAE) and Loss.

## Evaluation

Model performance is evaluated using MAE and Loss metrics. Plots visualize these metrics over epochs to assess model convergence and performance.

## Forecasting

Once trained, the model is used to forecast future maximum temperatures. Results are plotted alongside actual validation data to visualize the model's predictions.

## Dependencies

The project utilizes TensorFlow and its Keras API for model development and training. Other dependencies include pandas for data handling and matplotlib for plotting.

## Files

- `weather_model.ipynb`: Jupyter Notebook containing the complete code for data loading, preprocessing, model development, training, evaluation, and forecasting.
- `london_weather.csv`: Dataset containing historical maximum temperature records for London.

## Instructions

To run the project:
1. Ensure Python environment is set up with necessary libraries (TensorFlow, pandas, matplotlib).
2. Open and run the `weather_model.ipynb` notebook in a Jupyter environment.
3. Follow the code cells sequentially to load data, preprocess, build and train the model, evaluate performance, and generate forecasts.

## Conclusion

This project demonstrates the application of time series forecasting techniques to predict maximum temperatures. It highlights the importance of model architecture, data preprocessing, and evaluation metrics in building accurate weather prediction models.

