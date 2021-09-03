## How do we determine good _k_, the number of clusters?

As humans, upon seeing a dataset on a graph, one can infer the number of clusters intuitively and hence determine the appropriate _k_. Essentially what happens from a logical perspective in this process, is that we check for the relative **sparsity** between the datapoints and accordingly determine a good value for _k_.

Therefore, we could check for the relative distance (euclidean) of a point from another (across all such points) and see if one of these distances is much varied than the rest, giving us an idea of the varied point being farther away from the other points and hence possibly being part of a different cluster.

We then do this for all the points to provide an educated guess for the number of possible clusters, thus giving a good estimate for _k_.
