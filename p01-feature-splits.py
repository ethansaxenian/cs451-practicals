# Decision Trees: Feature Splits

# Python typing introduced in 3.5: https://docs.python.org/3/library/typing.html
from typing import List

# As of Python 3.7, this exists! https://www.python.org/dev/peps/pep-0557/
from dataclasses import dataclass

# My python file (very limited for now, but we will build up shared functions)
from shared import TODO


# Let's define a really simple class with two fields:
@dataclass
class DataPoint:
    temperature: float
    frozen: bool


data = [
    # vermont temperatures; frozen=True
    DataPoint(0, True),
    DataPoint(-2, True),
    DataPoint(10, True),
    DataPoint(11, True),
    DataPoint(6, True),
    DataPoint(28, True),
    # warm temperatures; frozen=False
    DataPoint(45, False),
    DataPoint(76, False),
    DataPoint(60, False),
    DataPoint(34, False),
    DataPoint(98.6, False),
]


def is_water_frozen(temperature: float) -> bool:
    """
    This is how we **should** implement it.
    """
    return temperature <= 32


# Make sure the data I invented is actually correct...
for d in data:
    assert d.frozen == is_water_frozen(d.temperature)


def find_candidate_splits(data: List[DataPoint]) -> List[float]:
    sorted_temps = sorted([point.temperature for point in data])  # isolate the temperatures into a sorted list
    midpoints = []
    for i in range(len(sorted_temps)-1):
        midpoints.append((sorted_temps[i] + sorted_temps[i+1]) / 2)
    return midpoints


def gini_impurity(points: List[DataPoint]) -> float:
    """
    The standard version of gini impurity sums over the classes:
    """
    p_ice = sum(1 for x in points if x.frozen) / len(points)
    p_water = 1.0 - p_ice
    return p_ice * (1 - p_ice) + p_water * (1 - p_water)
    # for binary gini-impurity (just two classes) we can simplify, because 1 - p_ice == p_water, etc.
    # p_ice * p_water + p_water * p_ice
    # 2 * p_ice * p_water
    # not really a huge difference.


def impurity_of_split(points: List[DataPoint], split: float) -> float:
    smaller = [point for point in points if point.temperature < split]
    bigger = [point for point in points if point.temperature > split]
    return gini_impurity(smaller) + gini_impurity(bigger)


if __name__ == "__main__":
    print("Initial Impurity: ", gini_impurity(data))
    print("Impurity of first-six (all True): ", gini_impurity(data[:6]))
    print("")
    for split in find_candidate_splits(data):
        print(f"splitting at {split} gives us impurity {impurity_of_split(data, split)}")
    test_data = [
        DataPoint(6, True),
        DataPoint(2, True),
        DataPoint(4, True),
        DataPoint(0, True),
        DataPoint(8, True),
    ]
    assert find_candidate_splits(test_data) == [1, 3, 5, 7]
