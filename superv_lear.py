from coordinate import *


with open('DataWrangling/df_features_train.pickle', 'rb') as f:
    df_train = pickle.load(f)
with open('DataWrangling/df_features_test.pickle', 'rb') as f:
    df_test = pickle.load(f)

#get training and testing data
X_train = df_train.drop(labels=['targets'], axis=1)
y_train = df_train['targets']

X_test = df_test.drop(labels=['targets'], axis=1)
y_test = df_test['targets']


def gaussiannb():
	#instantiate the classifier
	clf_NB = GaussianNB()

	#fit to training data
	clf_NB.fit(X_train, y_train)

	y_pred_NB = clf_NB.predict(X_test)

	accuracy_NB = np.mean(y_pred_NB == y_test)
	print("The accuracy of Naive Bayes is {0:.2%}".format(accuracy_NB))

	class_names=np.array(['bass', 'brass', 'flute', 'guitar', 
		     'keyboard', 'mallet', 'organ', 'reed', 
		     'string', 'synth_lead', 'vocal'])

	plot_confusion_matrix(y_test, y_pred_NB, classes=class_names, normalize=True,
		              title='Normalized confusion matrix for Naive Bayes')
	plt.savefig('ConfusionMatrix/NB_normalized.png')


def randomforestclassifier():
	#instantiate the random forest
	clf_Rf = RandomForestClassifier(n_estimators=20, max_depth=50, warm_start=True)

	clf_Rf.fit(X_train, y_train)

	y_pred_RF = clf_Rf.predict(X_test)

	accuracy_RF = np.mean(y_pred_RF == y_test)
	print("The accuracy of Random Forest is {0:.2%}".format(accuracy_RF))

	plot_confusion_matrix(y_test, y_pred_RF, classes=class_names, normalize=True,
		              title='Normalized confusion matrix for Random Forest')
	plt.savefig('ConfusionMatrix/RF_Normalized.png')


def gridsearch():
	param_dist = {"n_estimators" : [20, 40, 60, 80],
		      "max_depth": [10, 20, 30, 40],
		      "max_features": sp_randint(1, 11),
		      "min_samples_split": sp_randint(2, 11),
		      "bootstrap": [True, False],
		      "criterion": ["gini", "entropy"]}

	#instantiate a new random forest
	clf_RF_CV=RandomForestClassifier()

	#set number of iterations
	n_iter_search = 20
	#creat the random search class
	random_search_RF = RandomizedSearchCV(clf_RF_CV, param_distributions=param_dist,
		                              n_iter=n_iter_search, cv=5)
	#
	random_search_RF.fit(X_train, y_train)

	y_pred_RF_random = random_search_RF.predict(X_test)
	accuracy_RF_random = np.mean(y_pred_RF_random == y_test)
	print("The accuracy of Random Forest is {0:.2%}".format(accuracy_RF_random))

	plot_confusion_matrix(y_test, y_pred_RF_random, classes=class_names, normalize=True,
		              title='Normalized confusion matrix for Random Forest After Randomized Search')
	plt.savefig('ConfusionMatrix/RF_Normalized_RandomSearch.png')
	random_search_RF.best_estimator_

	# pickle the trained model
	with open("SavedModels/random_search_RF.pickle", mode='wb') as file:
	    pickle.dump(random_search_RF, file)


def svc_():
	#instantiate the sclaer
	scaler = MinMaxScaler()

	#scale the feature space
	X_train_scale = scaler.fit_transform(X_train)
	X_test_scale = scaler.fit_transform(X_test)

	#instatiate the  classifier
	clf_svm = SVC(C=0.1)

	clf_svm.fit(X_train_scale, y_train)

	y_pred_svm = clf_svm.predict(X_test)
	accuracy_svm = np.mean(y_pred_svm == y_test)
	print("The accuracy of SVMs is {0:.2%}".format(accuracy_svm))

	#display non normalized confusion matrix
	confusion_matrix(y_test, y_pred_svm)

	plot_confusion_matrix(y_test, y_pred_svm, classes=class_names, normalize=True,
		              title='Normalized confusion matrix for SVMs')
	plt.savefig('ConfusionMatrix/SVM_Normalized.png')

