import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC


NB_MODELS = ["BNB", "CNB", "MNB", "GNB"]


def process(data: pd.DataFrame):
    for synopsis in data["summary"]:
        synopsis.replace("(less)", "")
        synopsis.replace("(", "")
        synopsis.replace(")", "")
        synopsis.replace("\"", "")
        synopsis.replace("\'", "")
        synopsis.replace(".", "")
        synopsis.replace(",", "")
        synopsis.replace(":", "")
        synopsis.replace(";", "")
        synopsis.replace("?", "")
        synopsis.replace("!", "")

def u_tags(data: pd.DataFrame) -> list:
    tags = []
    for i in range(len(data["genre"])):
        t = data["genre"][i]
        if t not in tags:
            tags.append(t)
    return tags


# best values that i found are default, lowercasing makes it worse
def random_forest_cv(i, n=100, grams=(1,1), lowercase=False, stop=False, max_features=None):
    match grams:
        case (1,2):
            g = "Unigrams and Bigrams"
        case (2,2):
            g = "Bigrams"
        case _:
            g = "Unigrams"
    if stop:
        if max_features:
            cv = CountVectorizer(max_features=max_features, lowercase=lowercase, ngram_range=grams, stop_words='english')
            print("Test", str(i) + ". Random Forest. " + str(max_features), g + ". Stop Words Removed")
        else:
            cv = CountVectorizer(lowercase=lowercase, ngram_range=grams, stop_words='english')
            print("Test", str(i) + ". Random Forest. All", g, "Stop Words Removed")
    else:
        if max_features:
            cv = CountVectorizer(max_features=max_features, lowercase=lowercase, ngram_range=grams)
            print("Test", str(i) + ". Random Forest. " + str(max_features), g)
        else:
            cv = CountVectorizer(lowercase=lowercase, ngram_range=grams)
            print("Test:", str(i) + ". Random Forest. All", g)
    X = cv.fit_transform(database['summary'])
    y = database['genre'].values
    X_train, X, y_train, y = train_test_split(X, y, test_size=.2, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.5, stratify=y)

    clf = RandomForestClassifier(n_estimators=n)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_dev.toarray())
    print("accuracy:", accuracy_score(y_dev, y_pred))
    print()

    cv = X = y = X_train = y_train = clf = None


def nb_cv(i, grams=(1, 1), stop=False, t='CNB'):
    match grams:
        case (1, 2):
            g = "Unigrams and Bigrams"
        case (2, 2):
            g = "Bigrams"
        case _:
            g = "Unigrams"
    if stop:
        print("Test", str(i) + ". Naive Bayes (" + t + "). Stop Words Removed.", g)
        cv = CountVectorizer(max_features=6000, ngram_range=grams, stop_words='english')
    else:
        print("Test", str(i) + ". Naive Bayes (" + t + ").", g)
        cv = CountVectorizer(max_features=6000, ngram_range=grams)

    X = cv.fit_transform(database['summary'])
    y = database['genre'].values
    X_train, X, y_train, y = train_test_split(X, y, test_size=.2, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.5, stratify=y)

    match t:
        case "CNB":
            clf = ComplementNB()
        case "MNB":
            clf = MultinomialNB()
        case "BNB":
            clf = BernoulliNB()
        case _:
            clf = GaussianNB()

    clf.fit(X_train.toarray(), y_train)

    y_pred = clf.predict(X_dev.toarray())
    print("accuracy:", accuracy_score(y_dev, y_pred))
    print()
    cv = X = y = X_train = y_train = clf = None


def svc(i, c=1.0):
    cv = CountVectorizer(max_features=1000, stop_words='english', lowercase=True)
    print("Test", str(i) + ". SVM.  1000 Unigrams. Stop Words Removed. C =", c)
    X = cv.fit_transform(database['summary'])
    y = database['genre'].values
    X_train, X, y_train, y = train_test_split(X, y, test_size=.2, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.5, stratify=y)
    clf = SVC(C=c)
    clf.fit(X_train.toarray(), y_train)

    y_pred = clf.predict(X_dev.toarray())
    print("accuracy:", accuracy_score(y_dev, y_pred))
    print()
    cv = X = y = X_train = y_train = clf = None


def default_mnb():
    print("DEFAULT")
    cv = CountVectorizer()
    X = cv.fit_transform(database['summary'])
    y = database['genre'].values
    X_train, X, y_train, y = train_test_split(X, y, test_size=.2, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.5, stratify=y)
    clf = MultinomialNB()
    clf.fit(X_train.toarray(), y_train)

    y_pred = clf.predict(X_dev.toarray())
    print("accuracy:", accuracy_score(y_dev, y_pred))

    # matrix = plot_confusion_matrix(clf, X_dev, y_dev, cmap=plt.cm.Blues)
    # matrix.ax_.set_title('Baseline Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.gcf().axes[0].tick_params()
    # plt.gcf().axes[1].tick_params()
    # plt.show()

    return accuracy_score(y_dev, y_pred)


def tuned_mnb_v1(test=False, cf=False):
    cv = CountVectorizer(max_features=6000, stop_words='english', lowercase=True)
    bcv = CountVectorizer(max_features=1000, stop_words='english', ngram_range=(2,2))
    feats = FeatureUnion([('unigrams', cv), ('bigrams', bcv)])

    X = feats.fit_transform(database['summary'])
    y = database['genre'].values

    X_train, X, y_train, y = train_test_split(X, y, test_size=.2, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.5, stratify=y)
    clf = MultinomialNB()
    clf.fit(X_train.toarray(), y_train)


    # print(confusion_matrix(y_dev, y_pred, labels=clf.classes_))
    # print(classification_report(y_dev, y_pred))

    if cf:
        matrix = plot_confusion_matrix(clf, X_dev, y_dev, cmap=plt.cm.Blues)
        matrix.ax_.set_title('Tuned Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.gcf().axes[0].tick_params()
        plt.gcf().axes[1].tick_params()
        plt.show()

    if test:
        y_test_pred = clf.predict(X_test.toarray())
        print("TEST")
        print(classification_report(y_test, y_test_pred))
        return accuracy_score(y_test,y_test_pred)
    else:
        y_pred = clf.predict(X_dev.toarray())
        print("accuracy:", accuracy_score(y_dev, y_pred))

    return accuracy_score(y_dev, y_pred)


def tuned_mnb_v2(test=False, cf=False):
    cv = CountVectorizer(max_features=6000, stop_words='english', lowercase=True)
    bcv = CountVectorizer(max_features=1000, stop_words='english', ngram_range=(2,2))
    ccv = CountVectorizer(max_features=10000, stop_words='english', analyzer='char_wb', ngram_range=(4,4))
    feats = FeatureUnion([('unigrams', cv), ('bigrams', bcv), ('chars', ccv)])

    X = feats.fit_transform(database['summary'])
    y = database['genre'].values

    X_train, X, y_train, y = train_test_split(X, y, test_size=.2, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.5, stratify=y)
    clf = MultinomialNB()
    clf.fit(X_train.toarray(), y_train)


    # print(confusion_matrix(y_dev, y_pred, labels=clf.classes_))
    # print(classification_report(y_dev, y_pred))

    if cf:
        matrix = plot_confusion_matrix(clf, X_dev, y_dev, cmap=plt.cm.Blues)
        matrix.ax_.set_title('Tuned Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.gcf().axes[0].tick_params()
        plt.gcf().axes[1].tick_params()
        plt.show()

    if test:
        y_test_pred = clf.predict(X_test.toarray())
        print("TEST")
        print(classification_report(y_test, y_test_pred))
        return accuracy_score(y_test,y_test_pred)
    else:
        y_pred = clf.predict(X_dev.toarray())
        print("accuracy:", accuracy_score(y_dev, y_pred))

    return accuracy_score(y_dev, y_pred)


def tuned_mnb(test=False, cf=False):
    cv = CountVectorizer(max_features=6000, stop_words='english', lowercase=True)
    bcv = CountVectorizer(max_features=1000, stop_words='english', ngram_range=(2,2))
    ccv = CountVectorizer(max_features=10000, stop_words='english', analyzer='char_wb', ngram_range=(4,4))
    rares = CountVectorizer(max_features=300, stop_words='english', max_df=.3)
    feats = FeatureUnion([('unigrams', cv), ('bigrams', bcv), ('chars', ccv), ('rares', rares)])

    X = feats.fit_transform(database['summary'])
    y = database['genre'].values

    X_train, X, y_train, y = train_test_split(X, y, test_size=.2, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.5, stratify=y)
    clf = MultinomialNB()
    clf.fit(X_train.toarray(), y_train)


    # print(confusion_matrix(y_dev, y_pred, labels=clf.classes_))
    # print(classification_report(y_dev, y_pred))

    if cf:
        matrix = plot_confusion_matrix(clf, X_dev, y_dev, cmap=plt.cm.Blues)
        matrix.ax_.set_title('Tuned Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.gcf().axes[0].tick_params()
        plt.gcf().axes[1].tick_params()
        plt.show()

    if test:
        y_test_pred = clf.predict(X_test.toarray())
        print("TEST")
        print(classification_report(y_test, y_test_pred))
        return accuracy_score(y_test,y_test_pred)
    else:
        y_pred = clf.predict(X_dev.toarray())
        print("accuracy:", accuracy_score(y_dev, y_pred))

    return accuracy_score(y_dev, y_pred)


database = pd.read_csv("data1.csv")
process(database)
genres = u_tags(database)

# # CODE THAT PRINTS ACCURACY
ITER = 1
random_forest_cv(ITER) # 55%
ITER += 1
random_forest_cv(ITER, max_features=1000)
ITER += 1
random_forest_cv(ITER, grams=(1, 2), max_features=1000)
ITER += 1
svc(ITER)
ITER += 1
svc(ITER, c=0.5)
ITER += 1
svc(ITER, c=.1)
ITER += 1

for mod in NB_MODELS:
    nb_cv(ITER, grams=(1, 1), stop=True, t=mod)
    ITER += 1
    nb_cv(ITER, grams=(1, 1), stop=False, t=mod)
    ITER += 1
    nb_cv(ITER, grams=(1, 2), stop=True, t=mod)
    ITER += 1
    nb_cv(ITER, grams=(1, 2), stop=False, t=mod)
    ITER += 1
    nb_cv(ITER, grams=(2, 2), stop=True, t=mod)
    ITER += 1
    nb_cv(ITER, grams=(2, 2), stop=False, t=mod)
    ITER += 1

# avg = 0
# for _ in range(50):
#     avg += tuned_mnb(0.7, test=True)
#     print()

default_mnb()

x = tuned_mnb_v1(test=True)
print("test v1 acc:", x)

x = tuned_mnb_v2(test=True)
print("test v2 acc:", x)

x = tuned_mnb()
print("\ntest v3 acc:", x)

# print(avg/50)
# print(database.shape)

