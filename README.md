# Forecasting Stock Market Data Using Time Series and Artificial Neural Network Techniques

## Abstract

Financial data is complicated and many experts believe that there are some aspects of financials which simply can’t be predicted, the stock market in particular. However, research shows that there are machine learning methods which have the capability of making precise forecasts based on this financial data. The purpose of this analysis was to determine what machine learning techniques proved to be the most effective in forecasting stock market data by running various algorithms and recording which methods were capable of producing the most effective forecasts of this stock market information. The two techniques being examined within this study and applied to the stock market are Time Series Analysis and Artificial Neural Networks (ANN). Different methods to model the data within the time series analysis were then selected, those being simple forecasting, linear regression, and ARIMA modelling. Following, the outcomes of the various models were compared, and it was determined that the most effective method in forecasting this financial market, the Dow Jones Index, was linear regression within time series forecasting. However, more analyses will need to be conducted to achieve a definite answer. 

## Introduction

It is often thought that the stock market is an unpredictable, unforgiving and erratic battleground that is nearly impossible to tame or formularize. This idea has been consistently fed to players and onlookers within the market for years with the dominant academic theories backing this perpetuation. The most well known theory pushing forward this idea is the efficient-market hypothesis which demonstrates that the market is efficient and therefore, there is no need for any analysis, as due to this efficiency the market cannot be predicted and will not provide an above-average outcome for any investor, as it will always regress to the mean (Jiang, 2021). However, in recent years this idea has become increasingly challenged, which can in part be attributed to the advancement of machine learning and its potential impact on stock market dynamics (Jiang, 2021). Therefore, it would be advantageous to determine how effective different machine learning techniques perform on stock market data. 

Thus, for the purpose of this analysis, I will be comparing the use of multiple different machine learning techniques that have been applied using data collected from the Dow Jones Industrial Index, a stock market index following 30 of the biggest industrial companies in the United States, in order to determine which methods are the most effective in forecasting stock market data. The techniques chosen find their basis in recent literature and many of which are the most popular techniques found in recent years. The techniques being compared include a varying group of time series analysis methods including simple forecasting, linear regression, and ARIMA modelling techniques as well as artificial neural network (ANN) analysis. Both the time series techniques and ANN are commonly applied to the stock market and have a history of providing results. The analysis will begin with a brief introduction to the theory behind time series analysis, the theory behind ANN analysis and their relationships to the stock market to provide background. Following, exploratory data analysis will occur in which the data will be cleaned and manipulated. Next, the data will be split into training and testing groups to prepare for the machine learning algorithms where the various different techniques will be applied and models will be produced. These techniques will then be compared and discussed to gain a deeper understanding of their benefits and challenges.

## Literature Review

#### Time Series

Time series analysis has been a staple in the business and investing world for a long time, dating back to 1965 when an economist made the first claim that there was no correlation between past and future stock prices (Staffini, 2020). Time series analysis is a unique machine learning method because it not only collects data over a specified period but is able to demonstrate changes over this period (Tableau, 2022). Thus, time is an entirely extra data dimension which can be manipulated and understood. This quality is immensely useful within the finance world as it allows for intimate and lengthened trend and seasonality observations as well as produces the ability to forecast. (Tableau, 2022). Therefore, when analyzing the stock market, specifically the Dow Jones Index, we can use this technique of forecasting to make a prediction based on our historical data (Tableau, 2022). Furthermore, time series techniques also work to identify and subsequently understand any patterns that arise within the ever changing and fluctuating stock market (Tableau, 2022). One of the key items to observe when applying time series analysis to the stock market is how different variables shift and change over the same period (Hayes, 2021). It possible that no patterns or seasonality may arise within certain stocks or entire markets, however other individual stocks may be more susceptible to seasonal trends, such as bathing suit company doing poorly in the winter. As discussed, forecasting is looking for patterns and is under the assumption that “the future does not reserve any significative innovation compared to what we already can observe” (Staffini, 2020). However, we must tread lightly as the stock market exists in a state of instability where unexpected and deeply impactful events can occur at any moment, such as the 2008 recession (Staffini, 2020). Despite this and counter-intuitively, it may be getting easier to predict stock prices. This is because as the industry has developed stakeholders have been developing complex algorithms which automate their moves on the financial markets (Staffini, 2020). Time series techniques have been a great aide in this development.

Therefore, conducting a time series analysis on the Dow Jones Index stocks has the potential to produce major insights. One of the time series methods this analysis will be using is ARIMA modelling of which the basis is that future predictions are based on past values and that it is not actual values which are observed but the difference between values (Hayes, 2021). Hayes (2021) works to explain how past values impact and play a role in future values as illustrated by this quote “an investor using an ARIMA model to forecast stock prices would assume that new buyers and sellers of that stock are influenced by recent market transactions when deciding how much to offer or accept for the security.” The ARIMA technique is supported by previous literature such as Mondal et al. (2014) who used ARIMA modelling on 50 stocks on the National Stock Exchange (NSE) in India in which the ARIMA model had 85% accuracy in predicting stock prices and Adebiyi et al. (2014) who showed high accuracy in predicting the New York Stock Exchange (NYSE). The second time series method that will be used in this analysis is linear regression specified to time series analysis. Linear regression has historically been utilized in the stock market as a method of stock prediction and this form of linear regression may have been one of the first methods stock prediction methods ever used. At a time when computers were not accessible and statistics had to be done manually, this model could be used for prediction purposes as it works to fit a straight line through various data points in order to determine where the future data points land, thus linear regression. (Gururaj et al., 2019). In the present, linear regression is widely used for many purposes. Many studies like Guraj et al. (2019) conduct linear regression models first as a basis and a benchmark because of its reliability. Although linear regression has its pitfalls and drawbacks, it’s simplicity will always keep it in scope. Simple time series forecasting techniques will also be applied to the data in an attempt to understand the seasonality and trend potential the data may hold.

####	Artificial Neural Networks 

Artificial Neural Networks (ANN) is a technique based on “computational networks which attempt to simulate, in a gross manner, the decision process in networks of nerve cell (neurons) of the biological (human or animal) central nervous system.” (Graupe, 2013). It is of particular importance due to the fact that computationally it is very simple but can produce very complex mathematical problems (Graupe, 2013). Artificial neural networks also allow many different types of data to be input, be it non-linear or non-stationary, and is still capable of producing effective answers (Graupe, 2013). For these amongst many other reasons, ANN has become one of the most popular techniques used to try to predict stock market performance. A study conducted by Ruxanda and Badea (2014) showcased the applicability of ANN by performing an analysis on the Croatian Stock Market in which they find that they are able to successfully predict stock market prices. Similarly, Zavadzki et al (2020) performed an analysis comparing different ANN models on the Brazilian Stock Market and in which they found that they are able to successfully forecast this financial market even throughout the COVID-19 pandemic where uncertainty was extremely high. Finally, Selvamuthu et al. (2019) found that they were also successful in using artificial neural networks to correctly forecast a whole financial market, this one being the Indian Stock Market. These studies provide a strong basis for further study and display that the success of ANN is not based on a single economic market but is applicable worldwide.

## Methodology

The data set used for this study was taken from the UCI Machine Learning Repository and is titled “CNNpred: CNN-based stock market prediction using a diverse set of variables Data Set”  and was compiled by Hoseinzade and Haratizadeh (2019). This data contains a diverse range of variables as well as contains data for multiple different markets. For this paper, only the data set regarding the Dow Jones Industrial Average was used. This full data set regards the period from January 2010 to November 2017, however, for this experiment the size of the data set was decreased to only expand the timeframe of January 2011 to December 2016. The data set was then split into roughly 80% training and 20% testing groups as is required and custom by the time series and ANN techniques.  The percentages were then slightly altered to allow for the training data to contain the years 2011-2014 and the testing data the years 2015-2016. These years contain no missing data and therefore did not impede the study going forward. As it is daily data, each year only contains roughly 252 instances as the Stock Market is only open Monday to Friday and not on holidays. Thus, the training group contained 1006 instances across 12 variables and the testing group contained 503 instances across the same 12 variables. In total, the data set contains 1510 unique dates. 

####	Variables

There are 12 variables which will be examined and manipulated in this study, those variables are “Close”, “Date”, "Volume", "Return2", "Return3", "Return4", "ROC5", "Oil", "Gold", "Gas", "NYSEReturn", and  "DollarIndex". Choosing variables for such a study is complex as the variables used in predicting market behaviour are diverse and have many theories of which to apply. Thus, there are multiple categories into which financial data fit. These categories may be considered fundamental or technical (Vaisla and Bhatt, 2010). Fundamental analysis contains macroeconomic data and data related to companies such as cash flow, price to earnings, and dividend yields amongst other similar metrics (Vaisla and Bhatt, 2010). Technical analysis holds the idea that history repeats itself and is a pattern, thus market behaviour can be predicted by analyzing the past correlations between price and volume of stocks as well as analyzing patterns and trends (Vaisla and Bhatt, 2010). 
 
Of the 12 financial variables selected for this analysis the variable “Close” is of most interest as it acts as the predicted variable. This variable represents the price at which the stock market closed at on a given day Close is the variable we are looking at to predict because being able to foresee the closing price of a stock would allow investors and anyone involved in the market to make smarter financial decisions. 

The second variable Volume measures the amount of stock traded on that day. Return2, Return3, and Return4 variables all respectively reflect the return the stock had 2, 3, and 4 days prior (Hoseinzade et al., 2019). The ROC5 variable is the rate of change of the stock market from 5 days before, thus it measures the percentage which the price has changed from then to the current day (Hoseinzade et al., 2019). The variables Oil, Gold, and Gas are all commodities which are known to have an impact on stock prices as it is thought that they reflect the global market and thus act as good predictor variables (Hoseinzade and Haratizadeh, 2019). Thus, these variables reflect the relative change in oil, gold, and gas prices (Hoseinzade et al., 2019). NYSEReturn is a variable indicating the return of the New York stock exchange index, which is relevant as it can indicate what market influences could be occurring on elsewhere (Hoseinzade et al., 2019). However, this variable has the potential to be flawed as it is measuring many of the same aspects as the Dow Jones Index, potentially reflecting collinearity. The final variable is DollarIndex which reflects the exchange rate of the US dollar, the currency in which the Dow Jones operates. This is an essential variable as exchange rate has an influence on company profits which in turn has an influence on stock prices (Hoseinzade et al., 2019). These 12 variables were chosen as they represent a broad range of the different categories in which financial markets may be measured. There are many other variables which could be applicable and should be applied in future research. 

####	Correlations

<p align="center"><sub>Figure 1. Correlation matrix of the 12 variables used in the study</sub></p>
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/77642758/188866193-ef74fac1-eb66-433f-bb02-100b8ed2bf44.png"></p>

A correlation matrix was conducted on all variables excluding Date. This correlation test signified that all variables are independent. However, the highest correlation occurred between the variables NYSEReturn and Oil and was found to be significant. This correlation was positive 0.41 and is logical, as oil is expected to be a strong indicator in stock market price and NYSEReturn is measuring a stock market similar to the Dow Jones. The same positive significant correlation of 0.41 also occurred between ROC5 and Return4, meaning the rate of change from 5 days prior and the return from 4 days prior are somewhat related. The variable Close does not have a substantial correlation with any other variables thus, these variables are all acceptable. Therefore, all variables will be kept and the study will proceed.

####	Metrics

All of the models used in this analysis will be compared against one another based on the same metrics to allow for a fair comparison. These main metrics are RMSE and MAE. RMSE and MAE are the generally accepted metric used to measure the effectiveness of regression based models. This can be seen in the studies by Vaisla and Bhatt (2010), Zavadzki et al (2020), and Mondal et al. (2014).  As time series analysis does not require data normalization prior to calculations the RMSE and MAE will be normalized post calculations to allow an equal comparison between them and the already normalized ANN model. The formula being used to standardize these metrics is ‘RMSE or MAE / Max Dependent Variable  * 100’ with the max dependent variable of Close being 19974.62. R-squared and correlations will also be used to analyze specific models.

## Results

###	A. Time Series Analysis

The frequency for the time series models was 252. The variable being transformed is Close, which attempts to predict the future closing prices in 2015 and 2016. See Appendix A3 and Figure 2 for graphs displaying the trend and seasonal decomposition data. The data was not normalized prior to calculations in order to attempt to predict exact prices. 

 
<p align="center"> <sub>Figure 2. Visualization of the trend of the variable Close over time</sub> </p>
<p align="center"> <img width="400" alt="image" src="https://user-images.githubusercontent.com/77642758/188866312-6a877996-9093-41eb-9036-3be8bf16bffc.png"> </p>

#### i. Simple Forecasting Methods

The first analysis run on the data were simple time series forecasting methods. These methods are the average method, the naïve method, and the seasonal naïve method. The naïve method uses the value from the last observation and the seasonal naïve method uses the value from the previous season. The average method was ineffective on this data and produced high errors and a poor graph, it was thus abandoned. The naïve method (see Figure 3) produced the smallest RMSE while the seasonal naïve method (see Appendix A4) produced a higher RMSE (Table 1). 

<sub>Table 1. A summary table of the different metrics used to compare Simple Forecasting models</sub>

|Model |Raw RMSE |Raw MAE |Normalized RMSE |
| ---- | ----- | ----- | ----- |
| Naïve Method |793.1671 |559.6588 |0.085109976 |
| Seasonal Naïve Method |1241.509 |1110.549 |0.13321884 |

 
<p align="center"><sub>Figure 3. Forecast of naïve method model displaying predicted, trend, and actual data</sub></p>
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/77642758/188866346-29d553e9-aac3-4447-9978-b37a5f1851a4.png"></p>

#### ii. Linear Regression

Following, various linear regression techniques were applied. The first technique was trend linear regression (Figure 4). This method produced a moderate error of 0.169 and a high R-squared of  0.916 (Table 2). Next, a regression using trend plus season was applied (Appendix A5). This method produced a higher error than the linear regression model using only trend (Table 2). The R-squared values between the two models were similar.  See Box 1 for summary.

<sub>Table 2. A summary table of the different metrics used to compare Linear Regression models</sub>

| Model |Raw RMSE |Raw MAE |Normalized RMSE |Adjusted-R Squared |
| ---- | ----- | ----- | ------ | ------ |				
| Linear Regression 1 |	1576.681 |	1328.289 |	0.169184125 |	0.9163 |
| Linear Regression 2 |	1667.266 |	1395.311 |	0.178904255 |	0.915 |
| Linear Regression 3 |	1665.2 |	1393.739 |	0.178682565 |	0.915 |

 
<p align="center"><sub>Figure 4. Forecast of Linear Regression Model 1 displaying predicted, trend, and actual data</sub></p>
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/77642758/188866375-7d484a3a-0080-4009-bdc3-abb842912c44.png"></p>

<sub>Box 1. Summary of important comparison metrics of Linear Regression Model 1</sub>
```
> summary(close.trend)

Call:
tslm(formula = dj4.tstr[, 2] ~ trend)

Residuals:
     Min       1Q   Median       3Q      Max 
-1521.84  -397.34    56.44   374.43  1323.36 

Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept) 1.096e+04  3.539e+01   309.8   <2e-16 ***
trend       6.388e+00  6.089e-02   104.9   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 560.8 on 1004 degrees of freedom
Multiple R-squared:  0.9164,	Adjusted R-squared:  0.9163 
F-statistic: 1.101e+04 on 1 and 1004 DF,  p-value: < 2.2e-16
```

####	iii. Auto-Regressive Integrated Moving Average (ARIMA)

The final time series method applied was the Auto-Regressive Integrated Moving Average (ARIMA) method. The ARIMA model (Figure 5) produced a raw RMSE of 1879.299 and an MAE of 1629.6. The normalized RMSE is 0.201656237 (Table 3). See Box 2 for the values of the training metrics against the model metrics. 

 
<p align="center"><sub>Figure 5. Forecast of ARIMA Model displaying predicted, trend, and actual data</sub></p>
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/77642758/188866396-a9bb61f9-5833-41bf-aa04-cb276eb032d6.png"></p>

<sub>Table 3. A summary table of the RMSE and MAE metrics between training and testing data in ARIMA Model</sub>

| Model*	| RMSE |	MAE |
| ----- | ----- | ----- |
|Training | 	116.4019 |	84.63129 |
| Final Model |	1879.299 |	1629.6 |

<sub>Box 2. Summary data displaying the error metrics of the ARIMA Model</sub>
```
> summary(close.ari)
Series: dj4.tstr[, 2] 
ARIMA(1,1,1) with drift 

Coefficients:
          ar1     ma1   drift
      -0.8501  0.7956  6.1579
s.e.   0.0811  0.0926  3.5654

sigma^2 = 13603:  log likelihood = -6207.38
AIC=12422.75   AICc=12422.79   BIC=12442.4

Training set error measures:
                      ME     RMSE      MAE          MPE      MAPE       MASE       ACF1
Training set -0.02160532 116.4019 84.63129 -0.006094711 0.6206636 0.05247471 0.01322913
```

### B. Artificial Neural Networks

The target variable being predicted is again the variable Close which will be testing the years 2015 and 2016. As is required for ANN, the variables were then normalized in to make it possible to proceed to analysis. The variables were normalized using soft max and a lambda value equal to 5. See Appendix A6 for boxplot of normalized variables. 

#### i.	Neural Net Algorithms

The formula for the neural net algorithms is set to predict the variable Close based on the 10 predictor variables of Volume, Return2, Return3, Return4, ROC5, Oil, Gold, Gas, NYSEReturn, and DollarIndex. The first neural net algorithm contained zero hidden layers (see Appendix A7). It had a very poor performance in which correlation and adjusted R-squared were low (Table 4).  However, the RMSE of 0.2669626  and MAE of 0.2407253 were satisfactory. Now, further models will be explored in which the hidden layers are increased. A second model was run with 3 hidden layers which generated a higher correlation of 0.2302818 between predicted and actual values while the RMSE and MAE metrics stayed nearly the same as they had been in the previous model (Appendix A8 and A9). The R-squared of this model also increased slightly to 5% variance explained. Finally, a third model with even further hidden layers was run. This third and final model contained 6 hidden layers. This model proved to be the best model as it presented a higher correlation of 0.3190142 as well as slightly lower errors compared to the other two models (Table 4). See Figures 6 and 7 below for the visualization and scatterplot of the final neural net model. Finally, a variable importance calculation was run to determine which variables had the greatest influence. It was determined that NYSEReturn was the most significant variable in predicting the Close price (Appendix A10). 

<sub>Table 4. A summary table of ANN analysis of 3 different models and the 4 metrics being used within the comparison</sub>

|Model |	RMSE |	MAE |	R-squared  |	Correlation |
| ----- | ----- | ----- | ------ | ----  |
|Neural Net 1 |	0.2669626 |	0.2407253 |	0.01987 |	0.1476997 |
|Neural Net 2 |	0.2675611 |	0.2286912 |	0.05114 |	0.2302818 |
|Neural Net 3 |	0.2628307 |	0.2224987 |	0.09998 |	0.3190142 |


 
<p align="center"><sub>Figure 6.  Visualization of the hidden layers in neural net model 3</sub></p>
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/77642758/188866435-b93a2fb1-d147-44ed-90fa-d4a8fe3a6f87.png"></p>

<p align="center"><sub>Figure 7.  Scatterplot visualizing the actual versus predicted values in neural net model 3</sub></p>
<p align="center"><img width="400" alt="image" src="https://user-images.githubusercontent.com/77642758/188866468-ee52fb25-a021-46b0-a6a7-f63cae454d2a.png"></p>

<sub>Box 3. Neural net model 3 summary</sub>
```
> summary(dj11.lm2)

Call:
lm(formula = dj5.te$Close ~ dj10.pred2)

Residuals:
     Min       1Q   Median       3Q      Max 
-0.49816 -0.24478  0.03587  0.22909  0.61875 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  0.19739    0.04053   4.870 1.50e-06 ***
dj10.pred2   0.57535    0.07629   7.542 2.19e-13 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.2549 on 502 degrees of freedom
Multiple R-squared:  0.1018,	Adjusted R-squared:  0.09998 
F-statistic: 56.88 on 1 and 502 DF,  p-value: 2.19e-13
```

## Discussion

The various models will be compared within themselves in which one model will emerge the strongest. The best models from the 4 techniques will then be compared and analyzed against each other to try to determine the strongest model overall.

Within the simple forecasting methods, both the naïve and seasonal naïve methods produced satisfactory results. The naïve method however proved to be more effective as it has a smaller normalized RMSE of the two at 0.085. This means that the method which was most successful was the method that used the value of the last observation, the naïve method. This indicates that seasonality does not play a large role in predicting this set of stock market data because it is shown to be more effective to look at the data from more recent periods rather than to look at a season. This is logical because as seen in Figure 3 the mean is constantly moving upwards and does not follow a seasonal trend.  Thus, out of the simple forecasting techniques the naïve method is superior and can be compared with the other methods.

The two linear regression models produced results which were expected as they followed a similar function as the simple forecasting technique methods. Thus, it is reasonable to expect that the linear regression model which only uses trend would be more effective because as established the seasonality does not appear to have a strong connection to the future movement of this market. In fact, this occurrence is observed. The RMSE of linear regression model 1 using trend has a normalized RMSE value of 0.169 whereas the linear regression model 2 has a normalized RMSE value of 0.178, a full tenth higher. As R-squared is a useful metric in measuring basic regression that can be examined too, although both models have nearly identical values for this metric. Both display that the models account for around 91% of the variance occurring in the models, verifying that these models are of good quality. Ultimately, as is the expected case, the strongest linear regression model is linear regression model 1 using trend only. 

The ARIMA model produces a single best model based on its structure. The RMSE produced by the training data was a deviation of around $116 in the closing price while the RMSE on the final model resulted in a potential deviation of roughly $1879 in the closing price. This is an enormous gap especially when considering the normalized RMSE for the training set is 0.012490386 while the model’s normalized RMSE is 0.201656237. This is a nearly 20% increase in error. The difference in values of MAE is similar. This determines that this is a poorly fit model, as a good model should have similar values in the training and testing sets. Additionally, a potential price difference of $1763 would have anyone involved in the stock market frightened and not trusting of this algorithm.  

The artificial neural network models all produced surprising results when considering that the literature shows that ANN models are generally quite effective. While most of the models produced low RMSE and MAE metrics they were not as low as basic linear regression. While these values are still relatively low they fair poorly against the other models in this study. More concerningly, the R-squared values in all models are extremely low meaning that the independent variables do not significantly account for the variance in the dependent and predictor variable, Close. This could be occurring for many reasons, as discussed in the variables section there are many different types of financial metrics that can and should be considered in the predicting of stock market values. This lower R-squared value may indicate that key variables which would be strong predictors are missing. This is logical due to the fact that the RMSE metric is still quite low and RMSE accounts for how far apart the predicted and observed values are. The low R-squared value may be accounted for due to the large spread. With this in mind, we can consider that the third neural network model neural net model 3 is the best fit model for this set of data considering the slightly higher R-squared and lower errors. 
	
As the best models from each technique have now been selected it is necessary to compare all of these models to determine best fit. See Table 5 below for a summarized comparison of the normalized RMSE and MAE values as well as R-squared values for the models in which it applies. Upon first glance, it may be inferred that the naïve method results in the best model due to it having the lowest RMSE and MAE. However, it is very basic and not taking all factors into account but rather just using the previous days information to predict the next day. This could be effective but is not necessarily applicable when looking at a financial market because although it is more effective than looking at seasonality, large dips can occur between days due to unexpected events and it does not consider the overall trend. The ARIMA model is similarly likely not the best option. The variation between the training RMSE and final model RMSE is of great concern. The Dow Jones index is dealing in millions of dollars daily and having such a large difference in the price of the market between the training and testing set indicates this model was overfit. That leaves the final two models to be compared, either linear regression or neural networks. First, they have significantly different RMSE and MAE values. When considering that the scale they are being compared on is between 0 and 1, the difference between 0.17 and 0.26 is great. The linear regression model therefore has significantly lower error between its predicted and observed values, making it more reliable. People involved in the stock market like to be sure of their decisions and if one option is more reliable than the other it is an easy choice. The linear regression model is further embossed when considering its R-squared value compared to the neural network model. 91% versus 10% is a major difference. This means that the predictor variables in the linear regression model are accounting for 91% of the Close variable. This is quite reliable and would bring comfort to anybody investing in the stock market. Therefore, linear regression model 1 is the best fit model for the Dow Jones Index. 

<sub>Table 5. Metrics used to compare all models</sub>

| Algorithm | Model |	Normalized RMSE |	Normalized MAE |	Adjusted-R Squared |
| ----| ----- | ------ | ----- | ----- |
| Time Series  |	Naïve Method  |	0.085109976 |	0.060053609 |	NA |
| Time Series  | Linear Regression Model 1 |	0.169184125 |	0.142530678 |	0.9163 |
| Time Series  | ARIMA |	0.201656237 |	0.174862544 |	NA |
| ANN |	Neural Net Model 3 |	0.2628307 |	0.2224987 |	0.09998 |

## Conclusion

Attempting to measure the stock market has been of interest to scholars and investors for decades and there have been significant improvements in this area in the past years. The recent integration of machine learning within the stock market has only worked to ignite more interest and in turn, drive the need for more technological advancements of these techniques. When considering using machine learning for predicting the stock market there are two general techniques which are largely considered to yield the most predictable results, time series techniques and artificial neural networks. Ultimately, it was determined that the time series technique of linear regression was the most effective in predicting this market index. Linear regression resulted in the lowest errors and the highest account of variance. This may be unexpected due to the sophistication of other models like artificial neural networks; however, it is clear, the simplest can be best.

## References

Ariyo, A.A., Adewumi, O., Ayo, C. K., 2014. Stock Price Prediction Using the ARIMA Model. In: Institute of Electrical and Electronics Engineers, UKSim-AMSS 16th International Conference on Computer Modelling and Simulation. Washington, D.C., USA, 26 – 28 March 2014. IEEE Computer Society.

Brown, M.S., Pelosi, M.J., Dirska, H., 2013. Dynamic-radius species-conserving genetic algorithm for the financial forecasting of Dow Jones index stocks. Machine Learning and Data Mining in Pattern Recognition, pp.27–41.  Available at: https://link.springer.com/chapter/10.1007/978-3-642-39712-7_3 [Accessed 10 May 2022].

Ganti, A., 2022. Dow Jones Industrial Average (DJIA). Investopedia, [online] Available at: https://www.investopedia.com/terms/d/djia.asp [Accessed 21 April 2022].

Graupe, D., 2013. Principles Of Artificial Neural Networks. 3rd Edition. Singapore: World Scientific Publishing Company. 

Hayes, A., 2021. Autoregressive Integrated Moving Average (ARIMA). Investopedia, [online] Available at: https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp [Accessed 22 April 2022].

Hayes, A., 2021. What Is a Time Series? Investopedia, [online] Available at: https://www.investopedia.com/terms/t/timeseries.asp [Accessed 21 April 2022].

Hoseinzade, E., Haratizadeh, S., 2019. CNNpred: CNN-based stock market prediction using a diverse set of variables. Expert Systems with Applications, Available at: https://www.sciencedirect.com/science/article/pii/S0957417419301915 [Accessed 10 May 2022].

Hoseinzade, E., Haratizadeh, S., Khoeini, A., 2019. U-CNNpred: A Universal CNN-based Predictor for Stock Markets. arXiv preprint, Available at: https://arxiv.org/abs/1911.12540 [Accessed 10 May 2022].

Jiang, W., 2021. Applications of deep learning in stock market prediction: Recent progress. Expert Systems with Applications, Available at: https://www.sciencedirect.com/science/article/pii/S0957417421009441 [Accessed 11 May 2022].

Mondal, P., Shit, L., Goswami, S., 2014. Study Of Effectiveness Of Time Series Modeling (Arima) In Forecasting Stock Prices. International Journal of Computer Science Engineering and Applications, Available at: https://www.researchgate.net/publication/276197260_Study_of_Effectiveness_of_Time_Series_Modeling_Arima_in_Forecasting_Stock_Prices [Accessed 5 May 2022].

Ruxanda, G., Badea, L.M., 2014. Configuring Artificial Neural Networks for stock market predictions. Technological and Economic Development of Economy, Available at: https://www.researchgate.net/publication/261223631_Configuring_Artificial_Neural_Networks_for_stock_market_predictions [Accessed 11 May 2022].

Selvamuthu, D., Kumar, V., Mishra, A., 2019. Indian stock market prediction using artificial neural networks on tick data. Financial Innovation, Available at: https://jfin-swufe.springeropen.com/articles/10.1186/s40854-019-0131-7 [Accessed 10 May 2022].

Staffini, A., 2022. Stock Price Forecasting by a Deep Convolutional Generative Adversarial Network. Frontiers in Artificial Intelligence, Available at: https://www.frontiersin.org/articles/10.3389/frai.2022.837596/full [Accessed 11 May 2022].

Stoltzfus, J., 2022. How can a 'random walk' be helpful in machine learning algorithms? Techopedia, [online] Available at: https://www.techopedia.com/how-can-a-random-walk-be-helpful-in-machine-learning-algorithms/7/33166 [Accessed 22 April 2022].

Tableau, 2022. Time Series Analysis: Definition, Types, Techniques, and When It's Used. [Online] https://www.tableau.com/learn/articles/time-series-analysis [Accessed 21 April 2022]

Teixeira Zavadzki de Pauli, S., Kleina, M., Bonat, W.H., 2020. Comparing Artificial Neural Network Architectures for Brazilian Stock Market Prediction. Annals of Data Science, Available at: https://link.springer.com/article/10.1007/s40745-020-00305-w [Accessed 6 May 2022].

Vaisla, K.S., Bhatt, A.K., 2010. An analysis of the performance of artificial neural network technique for stock market forecasting. International Journal on Computer Science and Engineering, Available at: https://www.semanticscholar.org/paper/An-Analysis-of-the-Performance-of-Artificial-Neural-Vaisla-Bhatt/c478621873110a6cc5f5886705594d2c4cb6ea56 [Accessed 5 May 2022].

