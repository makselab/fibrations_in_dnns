
def make_algorithm(name, cfg, distance_thr):
    """
    Crea un algoritmo de clustering basado en el nombre y configuración.
    
    Args:
        name (str): Name of algorithm ('linkage_fcluster' o 'agg_clustering')
        cfg (dict): Configuration of the algorithm
        distance_thr (float): Distance threshold
    
    Returns:
        clustering function
    """
    
    # Available methods =======================================

    def linkage_fcluster_clusterer(distance):
        from scipy.cluster.hierarchy import linkage, fcluster
        linkage = cfg.get('linkage', 'average')
        
        Z = linkage(distance, method=linkage)
        clusters = fcluster(Z, t=distance_thr, criterion='distance')
        return clusters
    
    def agg_clustering_clusterer(distance):
        from sklearn.cluster import AgglomerativeClustering
        linkage = cfg.get('linkage', 'average')
        
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_thr,
            linkage=linkage, metric='precomputed')

        clusters = clustering.fit_predict(distance)
        return clusters

    algorithms = {
        'linkage_fcluster': linkage_fcluster_clusterer,
        'agg_clustering': agg_clustering_clusterer}
        
    return algorithms[name]