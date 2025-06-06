# =================================================================
# MODULES.

from pandas import read_csv
import argparse
from src.coloring.coloring import covering_layer

# =================================================================
# PARAMETERS - VARIABLES.

parser = argparse.ArgumentParser()
parser.add_argument('--resPATH', type=str)
args = parser.parse_args()

# 0,2,4 : Conv2d
# 6,8,10: Linear

# =================================================================
# PROCESSING.

for idx_LL in range(5):
	df_fibrations    = read_csv(args.resPATH + 'symmetries/fibrations/Layer_' + str(idx_LL) + '.csv',header=None)
	df_opfibrations = read_csv(args.resPATH + 'symmetries/optfibrations/Layer_' + str(idx_LL) + '.csv',header=None)

	df_cov = covering_layer(df_fibrations, df_opfibrations)
	df_cov.to_csv(args.resPATH + 'symmetries/covering/Layer_' + str(idx_LL) + '.csv', index = False, header= False)