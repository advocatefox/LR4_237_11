import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier

from tree_code import DecisionTree, find_best_split

mushrooms = pd.read_csv('data/agaricus-lepiota.data', header=None)
cars = pd.read_csv('data/car.data', header=None)
nursery = pd.read_csv('data/nursery.data', header=None)
students = pd.read_csv('data/students.csv')
ttt = pd.read_csv('data/tic-tac-toe-endgame.csv')

def encode(df):
    df = df.copy()
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


mushrooms_enc = encode(mushrooms)
cars_enc = encode(cars)
nursery_enc = encode(nursery)
ttt_enc = encode(ttt)
students_enc = encode(students)

def plot_surface(model, X, y, title=""):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.title(title)
    plt.show()

def run_default_tree(df, target_col, name):

    X = df.drop(columns=[target_col]).iloc[:, :2].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = DecisionTree()
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    print(f"\n===== {name} (CUSTOM TREE) =====")
    print("Train accuracy:", accuracy_score(y_train, train_pred))
    print("Test accuracy:", accuracy_score(y_test, test_pred))

    plot_surface(clf, X_train, y_train, f"{name} custom tree")

def sklearn_experiment(df, target_col, name):

    X = df.drop(columns=[target_col]).iloc[:, :2].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    params = [
        (None, 1),
        (3, 1),
        (5, 2),
        (10, 5)
    ]

    for depth, leaf in params:

        clf = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=42
        )

        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        print(f"\n{name} depth={depth}, leaf={leaf}")
        print("train:", train_acc, "test:", test_acc)

        plot_surface(clf, X_train, y_train, f"{name} depth={depth}")

def plot_gini_curves(df):

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    plt.figure(figsize=(10, 6))

    for col in X.columns:
        thresholds, ginis, _, _ = find_best_split(X[col], y)
        plt.plot(thresholds, ginis, label=str(col))

    plt.title("Gini curves")
    plt.legend()
    plt.show()


def scatter_students(df):

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    fig, axes = plt.subplots(1, len(X.columns), figsize=(20, 3))

    for i, col in enumerate(X.columns):
        axes[i].scatter(X[col], y, s=10)
        axes[i].set_title(col)


run_default_tree(mushrooms_enc, 0, "mushrooms")
run_default_tree(cars_enc, cars_enc.columns[-1], "cars")
run_default_tree(nursery_enc, nursery_enc.columns[-1], "nursery")
run_default_tree(ttt_enc, ttt_enc.columns[-1], "tic-tac-toe")
run_default_tree(students_enc, students_enc.columns[-1], "students")


sklearn_experiment(mushrooms_enc, 0, "mushrooms")
sklearn_experiment(cars_enc, cars_enc.columns[-1], "cars")


plot_gini_curves(students_enc)
scatter_students(students_enc)

m = mushrooms_enc
X = m.iloc[:, 1:].values
y = m.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

tree = DecisionTree()
tree.fit(X_train, y_train)

print("\nMUSHROOMS ACC:", accuracy_score(y_test, tree.predict(X_test)))

scorer = make_scorer(accuracy_score)

datasets = {
    "mushrooms": (mushrooms_enc, 0),
    "cars": (cars_enc, cars_enc.columns[-1]),
    "nursery": (nursery_enc, nursery_enc.columns[-1]),
    "ttt": (ttt_enc, ttt_enc.columns[-1]),
}

results = {}

for name, (df, target) in datasets.items():
    X = df.drop(columns=[target])
    y = df[target]

    model = DecisionTreeClassifier(random_state=42)

    scores = cross_val_score(model, X, y, cv=10, scoring=scorer)

    results[name] = scores.mean()

print("\nCROSS-VAL RESULTS:")
print(pd.DataFrame(results, index=["accuracy"]).T)