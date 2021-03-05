"""
In this lab, we'll go ahead and use the sklearn API to learn a decision tree over some actual data!

Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

We'll need to install sklearn.
Either use the GUI, or use pip:

    pip install scikit-learn
    # or: use install everything from the requirements file.
    pip install -r requirements.txt
"""

# We won't be able to get past these import statments if you don't install the library!
from sklearn.tree import DecisionTreeClassifier

import json  # standard python
from shared import dataset_local_path, TODO  # helper functions I made

# load up the data
examples = []
feature_names = set([])

with open(dataset_local_path("poetry_id.jsonl")) as fp:
    for line in fp:
        info = json.loads(line)
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["features"]
        # make a big list of all the features we have:
        for name in keep.keys():
            feature_names.add(name)
        # whether or not it's poetry is our label.
        keep["y"] = info["poetry"]
        # hold onto this single dictionary.
        examples.append(keep)

# Convert data to 'matrices'
# NOTE: there are better ways to do this, built-in to scikit-learn. We will see them soon.

# turn the set of 'string' feature names into a list (so we can use their order as matrix columns!)
feature_order = sorted(feature_names)

# Set up our ML problem:
train_y = []
train_X = []

# Put every other point in a 'held-out' set for testing...
test_y = []
test_X = []

for v, row in enumerate(examples):
    # grab 'y' and treat it as our label.
    example_y = row["y"]
    # create a 'row' of our X matrix:
    example_x = []
    for feature_name in feature_order:
        example_x.append(float(row[feature_name]))

    # put every fourth page into the test set:
    if v % 4 == 0:
        test_X.append(example_x)
        test_y.append(example_y)
    else:
        train_X.append(example_x)
        train_y.append(example_y)

print(f"There are {len(train_y)} training examples and {len(test_y)} testing examples.")


"""============== Experiments for Practical 02 ====================="""


def train_tree(param: str, value: any) -> tuple[float, float]:
    f = DecisionTreeClassifier()
    setattr(f, param, value)
    f.fit(train_X, train_y)

    return round(f.score(train_X, train_y), 3), round(f.score(test_X, test_y), 3)


def run_experiment(attr: str, values: list):
    training_data = {}
    testing_data = {}
    for v in values:
        train_score, test_score = train_tree(attr, v)
        training_data[train_score] = testing_data[test_score] = v
    best_training_score = max(training_data)
    best_testing_score = max(testing_data)
    print(f"best value for {attr} (training): {training_data[best_training_score]} with a score of {best_training_score}")
    print(f"best value for {attr} (testing): {testing_data[best_testing_score]} with a score of {best_testing_score}")


if __name__ == '__main__':
    experiments = {
        "splitter": ["best", "random"],
        "max_features": [*range(1, 20), "auto", "sqrt", "log2", None],
        "criterion": ["gini", "entropy"],
        "max_depth": [*range(1, 20), None],
        "random_state": [*range(1, 20), None]
    }
    for attribute, values in experiments.items():
        print("=" * 80)
        run_experiment(attribute, values)
