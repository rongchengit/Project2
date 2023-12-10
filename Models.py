#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

#in this section we are doing variable selection because we want to reduce the number of predictors not revelant to the variable ph
#this will improve model prediction performance for predicting ph
#make it a little easier to intrepret the model
#reduce the cost of computation because this is a business for ABC beverages
#in the print statements/graphs there are numbers and indicators that state which predictors are not strong correlators to ph and we are opting not to use them in our models

#Loading the Test Data for variable selection, model building, model and mode predicting
testingDataPath = r"C:\Users\Rongc\Desktop\Project2\TestingDataProcessed.csv"
testingData = pd.read_csv(testingDataPath)

#convert categorical variables to numeric using one-hot encoding
testingDataNumeric = pd.get_dummies(testingData)

#PH is the target variable our boss wants for ABC beverages
targetVariable = testingDataNumeric['ph']
features = testingDataNumeric.drop('ph', axis=1)

#calculating the correlation matrix for the numeric dataset
correlationMatrix = testingDataNumeric.corr()
phCorrelations = correlationMatrix['ph'].sort_values(ascending=False)

#selecting variables that are strong correlators to ph
#we used 0.5 and 0.3 because they are generally common for stats
correlationThresholdStrong = 0.5 # means theres a strong correlation
correlationThresholdModerate = 0.3 # means there is a moderate correlation
selectedFeaturesStrong = phCorrelations[(phCorrelations >= correlationThresholdStrong) | (phCorrelations <= -correlationThresholdStrong)].index
selectedFeaturesModerate = phCorrelations[(phCorrelations >= correlationThresholdModerate) & (phCorrelations < correlationThresholdStrong) | 
                                          (phCorrelations <= -correlationThresholdModerate) & (phCorrelations > -correlationThresholdStrong)].index
selectedFeaturesWeak = phCorrelations[abs(phCorrelations) < correlationThresholdModerate].index

#combining strong and moderate together
selectedFeatures = selectedFeaturesStrong.union(selectedFeaturesModerate).drop('ph')

#Prints of the actual value and the corresponding correlation to ph
print("Features with strong and moderate correlation to pH:")
for feature in selectedFeatures:
    correlation_value = phCorrelations[feature]
    correlation_category = "Strong" if abs(correlation_value) >= correlationThresholdStrong else "Moderate"
    print(f"{feature} ({correlation_category} Correlation): {correlation_value}")

print("\nFeatures with weak or no correlation to pH:")
for feature in selectedFeaturesWeak:
    if feature != 'ph':
        print(f"{feature}: {phCorrelations[feature]}")

#graph to show a visual of the correlation of variables correlating to ph
plt.figure(figsize=(15, 8))
sns.barplot(x=phCorrelations.index, y=phCorrelations.values)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Correlation with pH')
plt.title('Correlation of Features with pH')
plt.show()

#----------------------------------------------------------------
#model Building 
#in this section we are building multiple different models that comes with its own set of assumptions about the data
#it is for performance because different models can perform differently on various datasets. Some models might work well with a particular type of data distribution
#we chose linear, random forest, decision tree, Partial least squared, and polynomial 
#because linear serves as a good baseline for us as it's the simplest most easy to interpret 
#poly for the second easiest and allows non linear relationships
#random forest to capture relationships that are compelex in the data
#tree decision for non linear relationships
#pls for multicollinearity 
selectedFeaturesData = testingDataNumeric[selectedFeatures]

#splitting the dataset into training and testing sets
XTrain, XTest, YTrain, YTest = train_test_split(selectedFeaturesData, targetVariable, test_size=0.2, random_state=42)

#models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'PLSRegression': PLSRegression(n_components=2),
    'PolynomialRegression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
}

#training and evaluation data
modelStats = {}
for modelName, model in models.items():
    model.fit(XTrain, YTrain)
    predictions = model.predict(XTest)
    mse = mean_squared_error(YTest, predictions)
    rmse = np.sqrt(mse)  # Calculating RMSE
    r2 = r2_score(YTest, predictions)
    modelStats[modelName] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

#model performance
for modelName, stats in modelStats.items():
    print(f"{modelName} - MSE: {stats['MSE']:.3f}, RMSE: {stats['RMSE']:.3f}, R²: {stats['R2']:.3f}")


modelNames = list(modelStats.keys())
y_pos = np.arange(len(modelNames))
mseValues = [stats['MSE'] for stats in modelStats.values()]
rmseValues = [stats['RMSE'] for stats in modelStats.values()]
r2Values = [stats['R2'] for stats in modelStats.values()]

fig, ax = plt.subplots(1, 3, figsize=(18, 8))

ax[0].barh(y_pos, mseValues, align='center', color='skyblue')
ax[0].set_yticks(y_pos)
ax[0].set_yticklabels(modelNames)
ax[0].invert_yaxis()
ax[0].set_xlabel('MSE')
ax[0].set_title('Mean Squared Error (MSE) of Models')

ax[1].barh(y_pos, rmseValues, align='center', color='orange')
ax[1].set_yticks(y_pos)
ax[1].set_yticklabels(modelNames)
ax[1].invert_yaxis()
ax[1].set_xlabel('RMSE')
ax[1].set_title('Root Mean Squared Error (RMSE) of Models')

ax[2].barh(y_pos, r2Values, align='center', color='lightgreen')
ax[2].set_yticks(y_pos)
ax[2].set_yticklabels(modelNames)
ax[2].invert_yaxis()
ax[2].set_xlabel('R²')
ax[2].set_title('R-squared (R²) of Models')

plt.tight_layout()
plt.show()


#----------------------------------------------------------------
#selecting the model
#taking in the consideration for MSE, RMSE, and R2 out of all the models
#and comparing them to find the optimal model to predict ph
#polynomal model out performed the other models in all 3 key metrics together.
bestModelName = None
bestModelStats = {'MSE': np.inf, 'RMSE': np.inf, 'R2': -np.inf}

for modelName, stats in modelStats.items():
    if stats['MSE'] < bestModelStats['MSE'] and stats['RMSE'] < bestModelStats['RMSE'] and stats['R2'] > bestModelStats['R2']:
        bestModelName = modelName
        bestModelStats = stats

# Print the best model's performance
print(f"Best Performing Model: {bestModelName}")
print(f"MSE: {bestModelStats['MSE']:.3f}, RMSE: {bestModelStats['RMSE']:.3f}, R²: {bestModelStats['R2']:.3f}")

#----------------------------------------------------------------
#lastly we use the evaluation data with the best performing model for the prediction of PH
evaluationDataPath = r"C:\Users\Rongc\Desktop\Project2\EvaluationData.csv"
evaluationData = pd.read_csv(evaluationDataPath)
evaluationDataNumeric = pd.get_dummies(evaluationData)
evaluationDataFeatures = evaluationDataNumeric[selectedFeatures]

bestModel = models[bestModelName]
predictedPH = bestModel.predict(evaluationDataFeatures)

evaluationData['predictedPH'] = predictedPH
print(evaluationData[['predictedPH']])

# #exporting to excel
# outputCsvPath = r"C:\Users\Rongc\Desktop\Project2\PredictedPH.csv"
# evaluationData.to_csv(outputCsvPath, index=False)
# print(f"export done {outputCsvPath}")

#graph for ph
plt.figure(figsize=(10, 6))
plt.plot(evaluationData['predictedPH'].head(50), marker='o', linestyle='-', color='blue')
plt.title('Predicted pH Values')
plt.xlabel('Sample Index')
plt.ylabel('Predicted pH')
plt.grid(True)
plt.show()