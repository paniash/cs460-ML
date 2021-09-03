## How do we determine good k, the number of clusters?

As humans, upon seeing a dataset on a graph, one can infer the number of clusters intuitively and hence determine the appropriate 'k'. Essentially what happens in this process is that we check for the relative sparsity of the datapoints and accordingly determine a good 'k' value.

Hence, we could check for the relative distance (euclidean) of a point from another (across all such points) and see if one of these distances is much varied than the rest, giving us an idea of the varied point being farther away from the other points and hence possibly being part of a different cluster.
