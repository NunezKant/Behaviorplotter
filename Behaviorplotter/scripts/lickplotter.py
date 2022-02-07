import scipy.io
from tkinter import Tk   
from tkinter.filedialog import askopenfilename
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
sns.set_context("paper")
plt.style.use('bmh')

Tk().withdraw() 
filename = askopenfilename() 
data = scipy.io.loadmat(filename)

Licks = data['Licks']
RewardArray = np.array(data['Rewardposition'])
LickDf = pd.DataFrame(Licks, columns=['Trial','Position','Alpha','Rewarded','ts'])
LickDf['Position'] = LickDf['Position']*10
RewardArray = np.append(RewardArray.T,np.repeat(0,np.abs(LickDf['Trial'].max() - len(RewardArray.T))))


Reward_Durationdf = pd.DataFrame(RewardArray*10, columns=['Reward'])
Reward_Durationdf['Reward'].replace({0:np.nan}, inplace=True)


RewardedLicks_Df = LickDf[LickDf['Rewarded'] == 1]
NoRewardedLicks_Df = LickDf[LickDf['Rewarded'] == 0]

fig = plt.figure(figsize=(10, 10))
grid = plt.GridSpec(5, 5, hspace=0.2, wspace=0.2)

main_ax = fig.add_subplot(grid[1:,:4])
x_hist = fig.add_subplot(grid[:1, :4], yticklabels=[], sharex=main_ax)

main_ax.scatter(NoRewardedLicks_Df['Position'],NoRewardedLicks_Df['Trial'],label='no rewarded',s=5,alpha=0.5)
main_ax.scatter(RewardedLicks_Df['Position'],RewardedLicks_Df['Trial'], marker='*',label='rewarded',s=5,alpha=0.8)
main_ax.scatter(Reward_Durationdf['Reward'],Reward_Durationdf.index+1,s=20,marker='v', label='reward delivery')
main_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
main_ax.set_xlabel('Position (cm)')
main_ax.set_ylabel('Trial')
main_ax.set_xlim(-2,LickDf['Position'].max()+2)
sns.kdeplot(data=LickDf, x='Position',ax=x_hist,hue='Rewarded',legend=False)
x_hist.set_xlabel('')
sns.despine()
plt.show()