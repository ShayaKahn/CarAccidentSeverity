# Car accident severity prediction
I analyzed a dataset that contained information about accidents in the US. The name of the dataset is
'US_Accidents_Dec21_updated.csv'.
## Description of the dataset
### Features
| Feature | Description |
| ------- | ----------- |
| ID | This is a unique identifier of the accident record. |
| Start_Time | Shows the start time of the accident in the local time zone. |
| End_Time | Shows the end time of the accident in the local time zone. |
| Start_Lat | Shows latitude in the GPS coordinate of the start point. |
| Start_Lng | Shows longitude in GPS coordinate of the start point. |
| End_Lat | Shows latitude in GPS coordinate of the endpoint |
| End_Lng | Shows longitude in GPS coordinate of the endpoint. |
| Distance(mi) | The extent of the road affected by the accident. |
| Description | Shows natural language description of the accident. |
| Number | Shows the street number in the address field. |
| Street | Shows the street name in the address field. |
| Side | Shows the relative side of the street (Right/Left) in the address field. |
| City | Shows the city in the address field. |
| County | Shows the county in the address field. |
| State | Shows the state in the address field. |
| Zipcode | Shows the zipcode in the address field. |
| Country | Shows the country in the address field. |
| Timezone | Shows timezone based on the location of the accident (eastern, central, etc.). |
| Airport_Code | Denotes an airport-based weather station, which is the closest one to the location of the accident. |
| Weather_Timestamp | Shows the time-stamp of the weather observation record (in local time). |
| Temperature(F) | Shows the temperature (in Fahrenheit). |
| Wind_Chill(F) | Shows the wind chill (in Fahrenheit). |
| Humidity(%) | Shows the humidity (in percentage). |
| Pressure(in) | Shows the air pressure (in inches). |
| Visibility(mi) | Shows visibility (in miles). |
| Wind_Direction | Shows wind direction. |
| Wind_Speed(mph) | Shows wind speed (in miles per hour). |
| Precipitation(in) | Shows precipitation amount in inches, if there is any. |
| Weather_Condition | Shows the weather conditions (rain, snow, thunderstorm, fog, etc.). |
| Amenity | A Point-Of-Interest (POI) annotation that indicates the presence of an amenity in a nearby location. |
| Bump | A POI annotation which indicates the presence of a speed bump or hump in a nearby location. |
| Crossing | A POI annotation that indicates the presence of a crossing in a nearby location. |
| Give_Way | A POI annotation which indicates the presence of a give_way sign in a nearby location. |
| Junction | A POI annotation that indicates the presence of a junction in a nearby location. |
| No_Exit | A POI annotation which indicates the presence of a no_exit sign in a nearby location. |
| Railway | A POI annotation that indicates the presence of a railway in a nearby location. |
| Roundabout | A POI annotation which indicates the presence of a roundabout in a nearby location. |
| Station | A POI annotation that indicates the presence of a station (bus, train, etc.) in a nearby location. |
| Stop | A POI annotation that indicates the presence of a stop sign in a nearby location. |
| Traffic_Calming | A POI annotation that indicates the presence of traffic_calming means in a nearby location. |
| Traffic_Signal | A POI annotation which indicates presence of traffic_signal in a nearby location. |
| Turning_Loop | A POI annotation which indicates presence of turning_loop in a nearby location. |
| Sunrise_Sunset | Shows the period of day (i.e. day or night) based on sunrise/sunset. |
| Civil_Twilight | Shows the period of day (i.e. day or night) based on civil twilight. |
| Nautical_Twilight | Shows the period of day (i.e. day or night) based on nautical twilight. |
| Astronomical_Twilight | Shows the period of day (i.e. day or night) based on astronomical twilight. |

### Target
| Target | Description |
| ------- | ----------- |
| Severity | Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay). |

## Goal and methods
The goal is to predict the "Severity" of the accident. I used the classification algorithms: Random Forest Classifier, Gradient Boosting Classifier, XGBoost and artificial neural network.
This data set contains categorical, numerical, and boolean features.

There are several methods I will try to compare to deal with the imbalanced dataset. The methods are:

1. Random Under Sampling
2. Random Over Sampling
3. SMOTE Oversampling

### Notes:
- Before applying oversampling, I applied undersampling since the dataset has become very large.

## Evaluation metrics

I chose to use the macro-averaged F1 score to evaluate the performance of the models.

## Model Performance Comparison

| Resampling Method       | Random Forest | XGBoost |
|-------------------------|--------------|---------|
| Random Under Sampling  | 0.53         | 0.55    |
| Random Over Sampling   | 0.56         | 0.59    |
| SMOTE Over Sampling    | 0.56         | 0.58    |

## Conclusion

In this project, I analyzed the US_Accidents_Dec21_updated dataset to predict accident severity. I developed a preprocessing pipeline that normalizes features, encodes categorical variables, removes outliers, and engineers new features. Since the data is highly imbalanced, I experimented with three resampling techniques (random under-sampling, random over-sampling, and SMOTE) and compared Random Forest and XGBoost classifiers. To find optimal hyperparameters, I performed a randomized search using 50,000 training samples and five-fold cross-validation.

Among the tested configurations, XGBoost with random over-sampling gave the best performance. I suspect random under-sampling performed poorly due to significant data loss, while SMOTE struggled in high-dimensional space.

Next, I trained the best model (XGBoost with random over-sampling) using 1.5 million samples. The final macro-averaged F1 score reached 0.73, which is promising given the strong imbalance in the dataset. I then studied how increasing the number of training samples affects precision and recall for each severity class. The minority classes saw a drop in recall (indicating more false negatives), likely due to partial overfitting to the majority class, whereas precision improved for minority classes because of fewer false positives. Overall, the macro-averaged F1 score improved for all classes. I chose the macro F1 metric to balance precision and recall across classes.

Lastly, I attempted to use XGBoost with Dask on my local machine (an i7-1255U). Because the dataset fits comfortably into memory and I only have a single processor, I observed no performance benefits—distributing the workload added scheduling overhead without providing additional compute resources.