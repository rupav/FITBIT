# FITBIT
The goal of this project is to use data from various accelerometers to tell if they perform barbell lifts correctly or incorrectly (5 different ways).


given dataset is in a csv file, so made into a data frame using pandas.read_csv(' ')
df.shape is found to be (19622,159)
out of 100 features have araound 19000 values missing i.e. nan, so they are dropped using df.dropna(axis = 1,inplace = True)
'username' and 'unnamed: 0' columns too were dropped because of their insignificance in testing data
next we have used df.describe(include=['O']) for object type data only 3 columns came out of object type, 'classe','new_window' and 'cvtd_timestamp' .
classe is mapped to int type using preprocessing.LabelEncode().fit_transform 
final shape after cleaning data is (19622, 53)
RandomForestClassifier under sklearn.ensemble and DecisionTreeClassifier under sklearn.tree were imported for tranforming dataa into a graph
for score, k-fold cross-validation is udes, with k/cv = 10 for both classifiers


Conclusion: RandomForestClassifier was found better than DecisionTreeClassifier
