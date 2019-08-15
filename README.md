# Mlib

Library of oft-used functionalities for pandas, matplotlib, and data science in general.

This consists of three sections:

* [`plotHelpers`](#plothelpers) &mdash; nice-looking plot routines
* [`utility`](*utility) &mdash; all sorts of helpers that don't fit into a specific bucket
* [`stats`](*stats) &mdash; functions statistical in nature

## `plotHelpers`

### `plotConfusionMatrix()`

Makes pretty plots of confusion matrix data.
Especially nice: has counts, recall and precision variants. (See docstring.)

### `detailedHistogram()`

Used for cases where you want a bin for each value of your data (best if integers).

Suppose we have an array of integer values.
These are counted by values, and bins are constructed on the range [-0.5, max(values) + 0.5], so the frequency of every value is shown in its own bin.

### `plotValueCounts()`

For use with Pandas.DataFrame types.
Provide the name of a column containing a modest number of distinct values (e.g., categoricals), and this will create a bar chart showing the counts of each value. (Like a `detailedHistogram()`, except does not include 'empty' bins, and is most appropriate for categoricals.

### `dependencePlot()`

***Not completed***

A version of `shap.dependence_plot()`, which will have a provision for including indices of data values that the user wants to highlight (for local interpretability).

Note: Shapley values, originally from collective game theory, are a strong measure of a feature's contribution to a given target value.
These plots illustrate Shapley values over an entire test set.
