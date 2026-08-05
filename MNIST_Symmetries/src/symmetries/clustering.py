def make_algorithm(name, cfg, distance_thr):
    """
    Crea un algoritmo de clustering basado en el nombre y configuración.

    Args:
        name (str): Name of algorithm ('linkage_fcluster' or 'agg_clustering')
        cfg (dict): Configuration of the algorithm
        distance_thr (float): Distance threshold

    Returns:
        clustering function
    """

    def linkage_fcluster_clusterer(distance):
        from scipy.cluster.hierarchy import linkage, fcluster
        linkage_method = cfg.get('linkage', 'average')
        Z        = linkage(distance, method=linkage_method)
        clusters = fcluster(Z, t=distance_thr, criterion='distance')
        return clusters

    def agg_clustering_clusterer(distance):
        from sklearn.cluster import AgglomerativeClustering
        linkage_method = cfg.get('linkage', 'average')
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_thr,
            linkage=linkage_method,
            metric='precomputed',
        )
        clusters = clustering.fit_predict(distance)
        return clusters

    algorithms = {
        'linkage_fcluster': linkage_fcluster_clusterer,
        'agg_clustering':   agg_clustering_clusterer,
    }
    return algorithms[name]
