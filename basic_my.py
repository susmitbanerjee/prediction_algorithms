from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle


def print_scores(actual, prediction):
    print("accuracy is %f" % accuracy_score(actual, prediction))
    print('precision is %f' % precision_score(actual, prediction))
    print('recall is %f' % recall_score(actual, prediction))
    print('f1-score is %f' % f1_score(actual, prediction))
    print('confusion matrix: ')
    print(confusion_matrix(actual, prediction))


def save_model(classifier):
    pickle.dump(classifier, open('logistic_regression_classifier.pkl', 'wb'))
