from sklearn import linear_model, neighbors, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 11 datapoints of [height,weight,shoe size] in [CM, KG, US] 
# #6 is a male with more female like characteristics to make it more difficult for the models
x = [[178, 81, 10.5], [188, 93, 12], [173, 73, 9], [183, 86, 11], [180, 79, 10], [160, 60, 8], [168, 64, 8], [163, 61, 7], [170, 68, 9], [160, 58, 6.5], [165, 66, 7.5]]

# 11 datapoints of male/female corresponding to measurements in x
y = ["male", "male", "male", "male", "male", "male", "female", "female", "female", "female", "female"]


#trying 3 models to see which one fits best
#running 100 trials and averaging accuracys
logTotal, knnTotal, forestTotal = 0,0,0
for i in range(100):

    #training and test data to be used in all 3 models
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #Logistic regression
    log = linear_model.LogisticRegression()
    log.fit(x_train, y_train)
    y_pred = log.predict(x_test)
    logTotal += accuracy_score(y_test, y_pred)


    #K-Nearest Neighbors
    knn = neighbors.KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    knnTotal += accuracy_score(y_test, y_pred)


    #Random Forest
    forest = ensemble.RandomForestClassifier()
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    forestTotal += accuracy_score(y_test, y_pred)

#outputting average accuracy of each model to determine which performed best
print("Logistic Regression Accuracy: ", (logTotal / 100))
print("K-Nearest Neighbors Accuracy: ", (knnTotal / 100))
print("Random Forest Accuracy: ", (forestTotal / 100))