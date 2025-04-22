# =================================================================
# MODULES.

from coloring import covering_layer
import pandas as pd

# =================================================================
# PROCESSING.

for idx_LL in range(4):
	df_fibrations    = pd.read_csv('/home/osvaldo/Documents/STEM-AI/Project_PufferAI/results/symmetries/fibrations/Layer_' + str(idx_LL) + '.csv',header=None)
	df_opfibrations = pd.read_csv('/home/osvaldo/Documents/STEM-AI/Project_PufferAI/results/symmetries/opfibrations/Layer_' + str(idx_LL) + '.csv',header=None)

	df_cov = covering_layer(df_fibrations, df_opfibrations)

	df_cov.to_csv('/home/osvaldo/Documents/STEM-AI/Project_PufferAI/results/symmetries/covering/Layer_' + str(idx_LL) + '.csv', index = False, header= False)