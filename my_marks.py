import itertools
import operator

marks = [5, 4, 3.5, 4, 5, 3.5, 4.5, 4.5, 3.5, 5, 5, 4.5, 4.5]
wages = [6, 4, 4, 6, 6, 6, 5, 6, 5, 5, 5, 5, 6, ]

#marks = [4, 5, 4, 4, 4, 4.5, 3.5, 4, 4.5, 4, 4.5, 5, 4, 4, 4.5, 3.5, 4]
#wages = [10, 3, 2, 6, 2, 5, 2, 3, 4, 4, 4, 2, 5, 1, 4, 6, 4, ]

average = sum(list(itertools.starmap(operator.mul, zip(marks, wages))))/sum(wages)
print(average)