import matplotlib.pyplot as plt 
import numpy as np 

means = (1.62, 1.64, 1.74, 1.87, 2.03, 2.11)
rmse_err = (0.255, 0.31, 0.15, 0.07, 0.22, 0.07)

fig, ax = plt.subplots()
ax.barh([0, 1], means[:2], xerr=rmse_err[:2], align='center', ecolor='black', capsize=5)
ax.barh([2, 3, 4, 5], means[2:], xerr=rmse_err[2:], align='center', alpha=0.5, ecolor='black', capsize=5)
ax.set_yticks(range(6))
ax.set_yticklabels(['Persistent\nWeighted\nDirac', 'Persistent\nDirac', 'XGBoost', 'Multitask', 'RF', 'KRR'])
ax.set_xlabel('Test RMSE')
plt.savefig('test.png', dpi=200)