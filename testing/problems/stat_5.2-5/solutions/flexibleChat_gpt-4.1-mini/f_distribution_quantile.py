# filename: f_distribution_quantile.py
from scipy.stats import f

df1 = 8

df2 = 4
alpha = 0.01

quantile_value = f.ppf(alpha, df1, df2)

print(f"The 0.01 quantile of the F-distribution with df1={df1} and df2={df2} is: {quantile_value}")
