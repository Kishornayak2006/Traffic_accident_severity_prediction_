# Road Traffic Severity Prediction
![Python](https://img.shields.io/badge/Python-3.8.10-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Frontend](https://img.shields.io/badge/Framework-Streamlit-red)
![Deployment](https://img.shields.io/badge/Cloud-Heroku-purple)

<img width="800" alt="image" src="doc/theme.jpg">


## Web Application: 
This application is builded to check the accidents severity.



<img width="800" alt="image" src="doc/Mysuru.gif">







## Introduction
With the increasing of road traffic infrastructures, motor vehicles, drivers, and traffic flow, the role of road traffic in supporting and guiding economic and social development is becoming more and more obvious. As a result, road traffic safety has increasingly become a key issue in concerning the safety of people’s lives and property, as well as affecting the quality and efficiency of economic and social development. Road traffic accidents are the process of simultaneous damage of people or things, which caused by the coupling imbalance of dynamic and static factors such as human, vehicle, road, and environment. Therefore, it is necessary to study the influencing factors, as well as the classification and identification model of the severity of road traffic accident, so as to pave the way for improving the safety level of road traffic.

## Problem Statement: 
This is a multi-class classification problem where we are predicting the severity of accident :
* Minor
* Moderate
* Severe

based on the other 31 features.

## Description: 
This data set is [Road Traffic Accidents](https://www.kaggle.com/saurabhshahane/road-traffic-accidents) from Kaggle.


### Data consists:
* The Dataset was quite imbalanced with 10415 records with **minor** injury, 1743 records with **Moderate** injury and just 158 records with **severe** injury.


## Installation

* Clone this repository and check the ```requirements.txt```:
    ```shell
    git clone https://github.com/kishornayak2006/Traffic_accident_severity_prediction_
    cd Accident-Severity-Prediction
    pip install -r requirements.txt
    ```
* Simply run:    
    ```shell
    streamlit run app.py
    ```








