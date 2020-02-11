import numpy as np

# From Pithikos' answer
# https://stackoverflow.com/questions/869778/..
# ../populating-a-list-array-by-index-in-python#54981194
# to Populating a list/array by index in Python?
# https://stackoverflow.com/questions/869778/..
# ../populating-a-list-array-by-index-in-python#54981194)


class dynamiclist(list):
    """
    List not needing pre-initialization

    Example:
        l = dynamiclist()
        l[8][1] = 10
        l[9][1] = 20
        print(l)
    """

    def __setitem__(self, index, value):
        size = len(self)
        if index >= size:
            self.extend(dynamiclist() for _ in range(size, index + 1))

        list.__setitem__(self, index, value)

    def __getitem__(self, index):
        size = len(self)
        if index >= size:		# allow for dimensions > 1
            self.extend(dynamiclist() for _ in range(size, index + 1))

        return list.__getitem__(self, index)
