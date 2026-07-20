import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/exp_01/Full_Grid_dL_vs_thr.csv')

def pareto_frontier(df, col1="dL", col2="params", minimize=(True, True)):
    """
    Devuelve la máscara booleana de los puntos en la frontera de Pareto.
    
    Algoritmo O(n log n): ordena por col1, luego barre col2.
    """
    vals = df[[col1, col2]].to_numpy(dtype=np.float64)
    
    # Si maximizamos, negamos para convertir a minimización
    if not minimize[0]: vals[:, 0] = -vals[:, 0]
    if not minimize[1]: vals[:, 1] = -vals[:, 1]
    
    # Ordenar por col1 ascendente; desempatar por col2 ascendente
    order = np.lexsort((vals[:, 1], vals[:, 0]))
    sorted_vals = vals[order]
    
    # Barrido: un punto es Pareto si ningún punto anterior domina su col2
    is_pareto = np.zeros(len(df), dtype=bool)
    min_col2 = np.inf
    for i, (_, v2) in enumerate(sorted_vals):
        if v2 <= min_col2:
            is_pareto[order[i]] = True
            min_col2 = v2
    
    return is_pareto

pareto_mask = pareto_frontier(data, col1="dL", col2="reduction_pars_coll")
pareto_df = data[pareto_mask].sort_values("dL").reset_index(drop=True)

print(f"Puntos totales   : {len(data):,}")
print(f"Frontera Pareto  : {pareto_mask.sum():,} puntos")

fig, axs = plt.subplots(1,1)

axs.scatter(data['reduction_pars_coll'], data['dL'])
axs.plot(pareto_df["reduction_pars_coll"],pareto_df["dL"], "-", color="crimson", ms=4, lw=1.5, label="frontera Pareto")
axs.set_xlim(0,1)
axs.set_ylim(0,0.35)
axs.legend(); 
plt.tight_layout(); 
plt.show()
fig.savefig(PATH_RES + args.exp_name + '/Optimal_Pareto_dL.png',format='png')

pareto_df.to_csv('/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/exp_01/pareto_frontier.csv', index=False)