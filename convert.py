# convert
import pandas as pd
data = pd.io.stata.read_stata('./data/master_data_07142019_a.dta')
data.to_csv('./data/master_data_07142019_a.csv')