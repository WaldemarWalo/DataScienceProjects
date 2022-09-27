# Title: Detecting defected mechanical bearings

## Brief Summary:
- Unbalanced, binary classification of mechanical bearing based on features from accelerometer data and DC motor. Dataset with 10 million records and 14 features
- Feature engineering with clustering, grouping data by experiment and extracting distribution properties: sum, mean, median, skew, kurtosis, iqr and others
- Achieved roc_auc score of 0.9959 with an AdaBoost and RepeatStratifiedKfold. Clustered created features with GausianMixture to achieve an Adjusted RAND score of 0.76


## Description
The goal of this project is to detect defective mechanical bearings with features captured with the below experiment setup: 2 mechanical bearings are mounted on a shaft that is rotated by a DC motor. Each bearing has an accelerometer mounted that records movement on the x, y and z axis. Measurements are captured in 3 phases:
    1. the bearings are accelerated from 0 to 1500 rpm
    2. rpm is held at 1500 for 10 seconds 
    3. rpm is decreased to 250

## Dataset:
- this is a public Dataset available on [Kaggle](https://www.kaggle.com/datasets/isaienkov/bearing-classification)
- the dataset contains over 10m records for 112 unique experiments with the below features:
    - experiment_id - 112 unique experiments, 1 experiment for 1 bearing mounted in slot 2 (bearing_2_id)
    - bearing_1_id - ID of the first bearing, always set to 0 - so this slot is always housing a good bearing
    - bearing_2_id - ID of the second bearing, 112 unique values for 100 defective and 12 good bearings
    - timestamp - timestamp of when the data was captured - equal intervals
    - a1_x, a1_y, a1_z - data from 1st accelerometers connected to 1st bearing
    - a2_x, a2_y, a2_z - data from 2nd accelerometers connected to 2nd bearing
    - motor's rpm
    - frequency (Hz)
    - power usage in Watts (w)
    - status - binary labels classifying bearing_2 as: 0 - defective, 1 - good

## EDA findings:
- the Hz feature has the same information as rpm and will be dropped
- rpm feature has 40 invalid negative values that will be dropped
- additionally, there are 4599 entries where RPM is greater than 6000
- there are 17 instances where power (w) draw is above 3.5 W, which are treated as outliers and are removed
- analysing the accelerometer data distribution of the xyz axis for good and faulty bearings clearly shows that defective bearings have a higher variance, the distribution has more irregularities, and it is ofthen skewed
- this is also confirmed by the pairplot for the xyz axis for both types of bearings
- xyz variance is positively correlated with rpm
- not all data points start at the same time, rpm does not always start from 0, and the duration of the 3 phases of the experiment is not always the same

## Cross Validation:
- for this dataset, I've used RepeatedStratifiedKFold with groups as the binary target to ensure that each fold will have an equal number of positive and negative samples and prevent model bias

## Feature Engineering:
- I've created a dataset with 200 bearings, 100 unique faulty bearings from slot 2, 12 unique good bearings from slot 2
- I've added the 88 entries for bearing_id=0 from slot 1
- Clustering: I've used GaussianMixture to identify the 3 stages of the experiment by splitting the rpm and timestamp features
- the final dataset for training was created by grouping the data by experiment_id and extracting distribution properties: sum, mean, median, skew, kurtosis, iqr and a few other

## Modeling
- Initial CV score of the below models produced roc_auc score above 0.9
- shortlisted best classifiers :AdaBoostClassifier, CatBoostClassifier and LGBMClassifier produced score of 0.986
- Removing features based on feature_importance gave the best  roc_auc score of 0.995 with AdaBoost


## Clustering
- I've selected the 10 best features used during classification and supplemented them with additional 6 features selected with a custom implementation of Sequential Feature Selection to cluster the bearings using the Gaussian Mixture Model. This solution achieved an Adjusted RAND score of 0.76