from sklearn import tree

# 10 datapoints of [height,weight,shoe size] in [CM, KG, US]
x = [[178, 81, 10.5], [188, 93, 12], [173, 73, 9], [183, 86, 11], [180, 79, 10], [168, 64, 8], [163, 61, 7], [170, 68, 9], [160, 58, 6.5], [165, 66, 7.5]]

# 10 datapoints of male/female corresponding to measurements in x
y = ["male", "male", "male", "male", "male", "female", "female", "female", "female", "female"]

#declaring classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

#getting the models prediction for my measurements
prediction = clf.predict([[175, 68, 10]])

#getting the models prediciton for my girlfriend's measurements
prediction2 = clf.predict([[168, 53.5, 8.5]])

#outputting predictions
print(prediction)
print(prediction2)