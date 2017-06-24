"""
Privacy Bot - Text Classifier

Usage:
    classifier.py <true_positives>  <true_negatives>  (--url | --text)

Options:
    <true_positives>    Documents that are privacy policies (tar file)
    <true_negatives>    Documents that are NOT privacy policies (tar file)
    -h --help           Show help
"""

from docopt import docopt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np

from privacy_bot.analysis import dataset


#TODO: Run again the exhaustive parameters search using gridsearch
TEXT_CLF = Pipeline([
    ('tfidf', TfidfVectorizer(
        norm='l1',
        use_idf=True,
        stop_words='english',
        lowercase=True,)),
    ('clf', LogisticRegression(
        #  penalty='elasticnet',
        #  alpha=0.000001,
        #  n_iter=50
    ))
])


URL_CLF = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char',
        lowercase=True,
        use_idf=True,
        ngram_range=(3, 6))),
    ('clf', LogisticRegression())
])


def train_classifier(X, clf):
    clf.fit(X.data, X.target)
    return clf
    # print(timeit.timeit(
    #     'clf.predict(["http://cliqz.com/privacy"])',
    #     globals={'clf': clf},
    #     number=100
    # ))

    # print(clf.predict([
    #     'https://docs.python.org/3.6/library/timeit.html',
    #     'https://cliqz.com/privacy',
    #     'https://cliqz.com/terms/privacy-policy'
    # ]))


def eval_classifier(X, clf):
    # Testing Precision with 10 folds
    scores = cross_val_score(
        clf,
        X.data,
        X.target,
        n_jobs=-1,
        cv=StratifiedKFold(10),
        scoring=make_scorer(precision_score))
    print("Precision: ", np.mean(scores))
    print("Scores:", scores)


def load_dataset(positive_path, negative_path, target):
    assert target == 'url' or target == 'text'

    if target == 'url':
        X = dataset.load_urls(positive_path, negative_path)
        clf = URL_CLF
    elif target == 'text':
        X = dataset.load_text(positive_path, negative_path)
        clf = TEXT_CLF

    return X, clf


def main():
    args = docopt(__doc__)

    X, clf = load_dataset(
        positive_path=args['<true_positives>'],
        negative_path=args['<true_negatives>'],
        target='url' if args['--url'] else 'text'
    )

    eval_classifier(X=X, clf=clf)


if __name__ == "__main__":
    main()
