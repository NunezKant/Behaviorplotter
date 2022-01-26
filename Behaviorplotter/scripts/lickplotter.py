import scipy.io
from tkinter import Tk   
from tkinter.filedialog import askopenfilename
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

Tk().withdraw() 
filename = askopenfilename() 
data = scipy.io.loadmat(filename)
Licks = data['Licks']
LickDf = pd.DataFrame(Licks, columns=['Trial','Position','Alpha','Rewarded','ts'])
sns.jointplot(data=LickDf,x='Position',y='Trial',hue='Rewarded')
plt.ylim([0,LickDf.Trial.max()+20])
plt.show()