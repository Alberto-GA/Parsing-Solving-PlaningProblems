import matplotlib.pyplot as plt
#from matplotlib import ticker
import pickle
import numpy as np

#%%
with open('ElevatorsP1.pickle', 'rb') as f:
    elevatorsP1 = pickle.load(f)

with open('SkillTeachingP1.pickle', 'rb') as f:
    skillTeachingP1 = pickle.load(f)
    
with open('SysAdminP1.pickle', 'rb') as f:
    sysAdminP1 = pickle.load(f)

with open('CrossingTrafficP4.pickle', 'rb') as f:
    froggerP4 = pickle.load(f)
    
with open('NavigationP2.pickle','rb') as f:
    navigationP2 = pickle.load(f)
    
#%%

# Elevators-------------------------------------------------------------------

optimData = []
execData = []

optimData.append(elevatorsP1['UCT-UCB']['expanded'][10][0])
execData.append(elevatorsP1['UCT-UCB']['expanded'][10][1])

for variant in elevatorsP1['UCT-EBC']['simple'].keys():
    
    optimData.append(elevatorsP1['UCT-EBC']['simple'][variant][0])
    execData.append(elevatorsP1['UCT-EBC']['simple'][variant][1])
    
optimData.append(elevatorsP1['maxUCT-UCB'][100][0])
execData.append(elevatorsP1['maxUCT-UCB'][100][1])

for variant in elevatorsP1['maxUCT-EBC'].keys():
    
    optimData.append(elevatorsP1['maxUCT-EBC'][variant][0])
    execData.append(elevatorsP1['maxUCT-EBC'][variant][1])



optimMean = []
optimStdDev = []
bestOptim = 0.0
for  mean,dev in optimData:
    optimMean.append( abs(mean) )
    optimStdDev.append( dev )

bestOptim = min(optimMean)

execMean = []
execStdDev = []
bestExec = 0.0
for  mean,dev in execData:
    execMean.append( abs(mean) )
    execStdDev.append( dev )
    
bestExec = min(execMean)

ind = np.arange(8)      # the x locations for the groups
width = 0.35            # the width of the bars


fig = plt.figure()                               # Create a figure
ax = fig.add_axes([0, 0, 1, 1])                  # Add axes to teh figure
plt.title('Elevators')                           # Set a title
ax.set_ylim(0.0, 105)                          # Set y axis limits
ax.set_ylabel('|V(s0)|', labelpad=10)            # Set y axis label

ax.yaxis.set_tick_params(which='major', size=5, width=0.75,
                         direction='in', right='on')         # Format ticks
ax.xaxis.set_tick_params(which='major', size=5, width=0.75, 
                         direction='in', top='on')           # Format ticks

plt.setp(ax.get_xticklabels(), ha="right", rotation=30)      # 

ax.set_xticks(ind + width / 2)

ax.set_xticklabels( ('UCT', 'UCT-EBCmax', 'UCT-EBCpair','maxUCT','maxUCT-EBCmax', 'maxUCT-EBCpair', 
                     'maxUCT-EBCmax*', 'maxUCT-EBCpair*') )

series1 = ax.bar(ind, optimMean, width, color=[0.69,0.55,0.78], yerr= optimStdDev)
series2 = ax.bar(ind+width, execMean, width, color=[0.1,0.8,0.78], yerr= execStdDev)

my_lim = plt.xlim()
ax.plot([-1, 10], [bestExec,bestExec], linestyle='dashed', linewidth=0.75, color = 'black')
plt.xlim(my_lim)


ax.legend( (series1[0], series2[0]), ('Optimised','Executed'))

plt.savefig('elevatorsP1.png', dpi=300, transparent=False, bbox_inches='tight')
plt.savefig('elevatorsP1.pdf', transparent=False, bbox_inches='tight')

#%%

#SkillTeaching----------------------------------------------------------------

optimData = []
execData = []

optimData.append(skillTeachingP1['UCT-UCB']['simple'][100][0])
execData.append(skillTeachingP1['UCT-UCB']['simple'][100][1])

for variant in skillTeachingP1['UCT-EBC']['simple'].keys():
    
    optimData.append(skillTeachingP1['UCT-EBC']['simple'][variant][0])
    execData.append(skillTeachingP1['UCT-EBC']['simple'][variant][1])
    
optimData.append(skillTeachingP1['maxUCT-UCB'][10][0])
execData.append(skillTeachingP1['maxUCT-UCB'][10][1])

for variant in skillTeachingP1['maxUCT-EBC'].keys():
    
    optimData.append(skillTeachingP1['maxUCT-EBC'][variant][0])
    execData.append(skillTeachingP1['maxUCT-EBC'][variant][1])



optimMean = []
optimStdDev = []
bestOptim = 0.0

for  mean,dev in optimData:
    optimMean.append( abs(mean) )        
    optimStdDev.append( dev )

bestOptim = max(optimMean)
    
execMean = []
execStdDev = []
bestExec = 0.0
for  mean,dev in execData:
    execMean.append( abs(mean) )
    execStdDev.append( dev )

bestExec = max(execMean)


ind = np.arange(8)      # the x locations for the groups
width = 0.35            # the width of the bars


fig = plt.figure()                               # Create a figure
ax = fig.add_axes([0, 0, 1, 1])                  # Add axes to teh figure
plt.title('Skill Teaching')                           # Set a title
ax.set_ylim(20, 90)                          # Set y axis limits
ax.set_ylabel('V(s0)', labelpad=10)            # Set y axis label

ax.yaxis.set_tick_params(which='major', size=5, width=0.75,
                         direction='in', right='on')         # Format ticks
ax.xaxis.set_tick_params(which='major', size=5, width=0.75, 
                         direction='in', top='on')           # Format ticks

plt.setp(ax.get_xticklabels(), ha="right", rotation=30)      # 

ax.set_xticks(ind + width / 2)

ax.set_xticklabels( ('UCT', 'UCT-EBCmax', 'UCT-EBCpair','maxUCT','maxUCT-EBCmax', 'maxUCT-EBCpair', 
                     'maxUCT-EBCmax*', 'maxUCT-EBCpair*') )



series1 = ax.bar(ind, optimMean, width, color=[0.69,0.55,0.78], yerr= optimStdDev)
series2 = ax.bar(ind+width, execMean, width, color=[0.1,0.8,0.78], yerr= execStdDev)

my_lim = plt.xlim()
ax.plot([-1, 10], [bestExec,bestExec], linestyle='dashed', linewidth=0.75, color = 'black')
plt.xlim(my_lim)

ax.legend( (series1[0], series2[0]), ('Optimised','Executed'))

plt.savefig('skillTeachingP1.png', dpi=300, transparent=False, bbox_inches='tight')
plt.savefig('skillTeachingP1.pdf', transparent=False, bbox_inches='tight')

#%%

# SysAdminP1 ----------------------------------------------------------------

optimData = []
execData = []

optimData.append(sysAdminP1['UCT-UCB']['simple'][100][0])
execData.append(sysAdminP1['UCT-UCB']['simple'][100][1])

for variant in sysAdminP1['UCT-EBC']['simple'].keys():
    
    optimData.append(sysAdminP1['UCT-EBC']['simple'][variant][0])
    execData.append(sysAdminP1['UCT-EBC']['simple'][variant][1])
    
optimData.append(sysAdminP1['maxUCT-UCB'][10][0])
execData.append(sysAdminP1['maxUCT-UCB'][10][1])

for variant in sysAdminP1['maxUCT-EBC'].keys():
    
    optimData.append(sysAdminP1['maxUCT-EBC'][variant][0])
    execData.append(sysAdminP1['maxUCT-EBC'][variant][1])



optimMean = []
optimStdDev = []
bestOptim = 0.0

for  mean,dev in optimData:
    optimMean.append( abs(mean) )        
    optimStdDev.append( dev )

bestOptim = max(optimMean)
    
execMean = []
execStdDev = []
bestExec = 0.0
for  mean,dev in execData:
    execMean.append( abs(mean) )
    execStdDev.append( dev )

bestExec = max(execMean)


ind = np.arange(8)      # the x locations for the groups
width = 0.35            # the width of the bars


fig = plt.figure()                               # Create a figure
ax = fig.add_axes([0, 0, 1, 1])                  # Add axes to teh figure
plt.title('Sys-Admin')                           # Set a title
ax.set_ylim(100, 400)                          # Set y axis limits
ax.set_ylabel('V(s0)', labelpad=10)            # Set y axis label

ax.yaxis.set_tick_params(which='major', size=5, width=0.75,
                         direction='in', right='on')         # Format ticks
ax.xaxis.set_tick_params(which='major', size=5, width=0.75, 
                         direction='in', top='on')           # Format ticks

plt.setp(ax.get_xticklabels(), ha="right", rotation=30)      # 

ax.set_xticks(ind + width / 2)

ax.set_xticklabels( ('UCT', 'UCT-EBCmax', 'UCT-EBCpair','maxUCT','maxUCT-EBCmax', 'maxUCT-EBCpair', 
                     'maxUCT-EBCmax*', 'maxUCT-EBCpair*') )



series1 = ax.bar(ind, optimMean, width, color=[0.69,0.55,0.78], yerr= optimStdDev)
series2 = ax.bar(ind+width, execMean, width, color=[0.1,0.8,0.78], yerr= execStdDev)

my_lim = plt.xlim()
ax.plot([-1, 10], [bestExec,bestExec], linestyle='dashed', linewidth=0.75, color = 'black')
plt.xlim(my_lim)

ax.legend( (series1[0], series2[0]), ('Optimised','Executed'))

plt.savefig('sysAdminP1.png', dpi=300, transparent=False, bbox_inches='tight')
plt.savefig('sysAdminP1.pdf', transparent=False, bbox_inches='tight')

#%%

# Navigation -----------------------------------------------------------------

optimData = []
execData = []

optimData.append(navigationP2['UCT-UCB']['expanded'][1][0])
execData.append(navigationP2['UCT-UCB']['expanded'][1][1])

for variant in navigationP2['UCT-EBC']['expanded'].keys():
    
    optimData.append(navigationP2['UCT-EBC']['expanded'][variant][0])
    execData.append(navigationP2['UCT-EBC']['expanded'][variant][1])
    
optimData.append(navigationP2['maxUCT-UCB'][100][0])
execData.append(navigationP2['maxUCT-UCB'][100][1])

for variant in navigationP2['maxUCT-EBC'].keys():
    
    optimData.append(navigationP2['maxUCT-EBC'][variant][0])
    execData.append(navigationP2['maxUCT-EBC'][variant][1])



optimMean = []
optimStdDev = []
bestOptim = 0.0

for  mean,dev in optimData:
    optimMean.append( abs(mean) )        
    optimStdDev.append( dev )

bestOptim = min(optimMean)
    
execMean = []
execStdDev = []
bestExec = 0.0
for  mean,dev in execData:
    execMean.append( abs(mean) )
    execStdDev.append( dev )

bestExec = min(execMean)


ind = np.arange(8)      # the x locations for the groups
width = 0.35            # the width of the bars


fig = plt.figure()                               # Create a figure
ax = fig.add_axes([0, 0, 1, 1])                  # Add axes to teh figure
plt.title('Navigation')                           # Set a title
ax.set_ylim(0, 50)                          # Set y axis limits
ax.set_ylabel('|V(s0)|', labelpad=10)            # Set y axis label

ax.yaxis.set_tick_params(which='major', size=5, width=0.75,
                         direction='in', right='on')         # Format ticks
ax.xaxis.set_tick_params(which='major', size=5, width=0.75, 
                         direction='in', top='on')           # Format ticks

plt.setp(ax.get_xticklabels(), ha="right", rotation=30)      # 

ax.set_xticks(ind + width / 2)

ax.set_xticklabels( ('UCT', 'UCT-EBCmax', 'UCT-EBCpair','maxUCT','maxUCT-EBCmax', 'maxUCT-EBCpair', 
                     'maxUCT-EBCmax*', 'maxUCT-EBCpair*') )



series1 = ax.bar(ind, optimMean, width, color=[0.69,0.55,0.78], yerr= optimStdDev)
series2 = ax.bar(ind+width, execMean, width, color=[0.1,0.8,0.78], yerr= execStdDev)

my_lim = plt.xlim()
ax.plot([-1, 10], [bestExec,bestExec], linestyle='dashed', linewidth=0.75, color = 'black')
plt.xlim(my_lim)

ax.legend( (series1[0], series2[0]), ('Optimised','Executed'))

plt.savefig('navigationP2.png', dpi=300, transparent=False, bbox_inches='tight')
plt.savefig('navigationP2.pdf', transparent=False, bbox_inches='tight')

#%%

# Frogger -----------------------------------------------------------------

optimData = []
execData = []

optimData.append(froggerP4['UCT-UCB']['simple'][1][0])
execData.append(froggerP4['UCT-UCB']['simple'][1][1])

for variant in froggerP4['UCT-EBC']['simple'].keys():
    
    optimData.append(froggerP4['UCT-EBC']['simple'][variant][0])
    execData.append(froggerP4['UCT-EBC']['simple'][variant][1])
    
optimData.append(froggerP4['maxUCT-UCB'][100][0])
execData.append(froggerP4['maxUCT-UCB'][100][1])

for variant in froggerP4['maxUCT-EBC'].keys():
    
    optimData.append(froggerP4['maxUCT-EBC'][variant][0])
    execData.append(froggerP4['maxUCT-EBC'][variant][1])



optimMean = []
optimStdDev = []
bestOptim = 0.0

for  mean,dev in optimData:
    optimMean.append( abs(mean) )        
    optimStdDev.append( dev )

bestOptim = min(optimMean)
    
execMean = []
execStdDev = []
bestExec = 0.0
for  mean,dev in execData:
    execMean.append( abs(mean) )
    execStdDev.append( dev )

bestExec = min(execMean)


ind = np.arange(8)      # the x locations for the groups
width = 0.35            # the width of the bars


fig = plt.figure()                               # Create a figure
ax = fig.add_axes([0, 0, 1, 1])                  # Add axes to teh figure
plt.title('Crossing Traffic')                           # Set a title
ax.set_ylim(0, 50)                          # Set y axis limits
ax.set_ylabel('|V(s0)|', labelpad=10)            # Set y axis label

ax.yaxis.set_tick_params(which='major', size=5, width=0.75,
                         direction='in', right='on')         # Format ticks
ax.xaxis.set_tick_params(which='major', size=5, width=0.75, 
                         direction='in', top='on')           # Format ticks

plt.setp(ax.get_xticklabels(), ha="right", rotation=30)      # 

ax.set_xticks(ind + width / 2)

ax.set_xticklabels( ('UCT', 'UCT-EBCmax', 'UCT-EBCpair','maxUCT','maxUCT-EBCmax', 'maxUCT-EBCpair', 
                     'maxUCT-EBCmax*', 'maxUCT-EBCpair*') )



series1 = ax.bar(ind, optimMean, width, color=[0.69,0.55,0.78], yerr= optimStdDev)
series2 = ax.bar(ind+width, execMean, width, color=[0.1,0.8,0.78], yerr= execStdDev)

my_lim = plt.xlim()
ax.plot([-1, 10], [bestExec,bestExec], linestyle='dashed', linewidth=0.75, color = 'black')
plt.xlim(my_lim)

ax.legend( (series1[0], series2[0]), ('Optimised','Executed'))

plt.savefig('CrossingTrafficP4.png', dpi=300, transparent=False, bbox_inches='tight')
plt.savefig('CrossingTrafficP4.pdf', transparent=False, bbox_inches='tight')