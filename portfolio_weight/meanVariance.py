__author__ = 'zoulida'

# -*- coding: utf-8 -*-


#构建一个MeanVariance类,该类可以根据输入的收益率序列,求解二次规划问题,计算出最优资产比例,并绘制最小方差前缘曲线
#定义 MeanVariance类
from matplotlib import pyplot as plt
from scipy import linalg
import numpy as np
import ffn
import cvxopt
from cvxopt import matrix



import ffn
from scipy import linalg
class MeanVariance:
    def __init__(self,returns):#定义构造器,传入收益率数据(dataframe格式的每日收益率)
        self.returns=returns
    def minVar(self,goalRet): #定义最小化方差的函数,即求解二次规划
        covs=np.array(self.returns.cov())
        means=np.array(self.returns.mean())
        L1=np.append(np.append(covs.swapaxes(0,1),[means],0),
                     [np.ones(len(means))],0).swapaxes(0,1)
        L2=list(np.ones(len(means)))
        L2.extend([0,0])
        L3=list(means)
        L3.extend([0,0])
        L4=np.array([L2,L3])
        L=np.append(L1,L4,0)
        results=linalg.solve(L,np.append(np.zeros(len(means)),[1,goalRet],0))
        return(np.array([list(self.returns.columns),results[:-2]]))

    def frontierCurve(self):#定义绘制最小方差前缘曲线函数
        goals=[x/500000 for x in range(-100,4000)]
        variances=list(map(lambda x: self.calVar(self.minVar(x)[1,:].astype(np.float)),goals))
        plt.plot(variances,goals)
        plt.show()

    def meanRet(self,fracs):#给定各资产的比例,计算收益率的均值
        meanRisky=ffn.to_returns(self.returns).mean()
        assert len(meanRisky)==len(fracs), 'Length of fractions must be equal to number of assets'
        return(np.sum(np.multiply(meanRisky,np.array(fracs))))

    def calVar(self,fracs): #给定各资产的比例,计算收益率方差
        return(np.dot(np.dot(fracs,self.returns.cov()),fracs))

    def solve_quadratic_problem(self,goal):
        covs = np.array(self.returns.cov())
        means = np.array(self.returns.mean())
        P = matrix(np.dot(2,covs))
        Q = matrix(np.zeros((len(means),1)))
        G = -matrix(np.zeros((len(means),len(means))))
        A = matrix(np.append([np.ones(len(means))],[means],axis=0))
        h = matrix(np.zeros((len(means),1)))
        b = matrix(np.array([[1,goal]]).swapaxes(0,1))
        sol = cvxopt.solvers.qp(P, Q, G , h, A, b)
        return sol


import pandas as pd
stock=pd.read_table('stock.txt',sep='\t',index_col='Trddt')
stock.index=pd.to_datetime(stock.index)
fjgs=stock.ix[stock.Stkcd==600033,'Dretwd']
fjgs.name='fjgs'
zndl=stock.ix[stock.Stkcd==600023,'Dretwd']
zndl.name='zndl'
sykj=stock.ix[stock.Stkcd==600183,'Dretwd']
sykj.name='sykj'
hxyh=stock.ix[stock.Stkcd==600015,'Dretwd']
hxyh.name='hxyh'
byjc=stock.ix[stock.Stkcd==600004,'Dretwd']
byjc.name='byjc'


sh_return=pd.concat([byjc,fjgs,hxyh,sykj,zndl],axis=1)
sh_return.head()

sh_return=sh_return.dropna()
#sh_return2=sh_return2.dropna()
sh_return.corr()


cumreturn=(1+sh_return).cumprod()
sh_return.plot()
plt.title('Daily Return of 5 Stocks(2014-2015)')
plt.legend(loc='lower center',bbox_to_anchor=(0.5,-0.3),
           ncol=5, fancybox=True, shadow=True)

cumreturn.plot()
plt.title('Cumulative Return of 5 Stocks(2014-2015)')
sh_return.corr()
plt.show()

minVar=MeanVariance(sh_return)

minVar.frontierCurve()

train_set=sh_return['2014']
test_set=sh_return['2015']
varMinimizer=MeanVariance(train_set)
goal_return=0.003 #目标收益设置为固定值
portfolio_weight=varMinimizer.minVar(goal_return)
print('portfolio_weight =', portfolio_weight)

test_return=np.dot(test_set,
                   np.array([portfolio_weight[1,:].astype(np.float)]).swapaxes(0,1))
test_return=pd.DataFrame(test_return,index=test_set.index)
test_cum_return=(1+test_return).cumprod()

sim_weight=np.random.uniform(0,1,(100,5))
sim_weight=np.apply_along_axis(lambda x: x/sum(x),1,sim_weight)
sim_return=np.dot(test_set,sim_weight.swapaxes(0,1))
sim_return=pd.DataFrame(sim_return,index=test_cum_return.index)
sim_cum_return=(1+sim_return).cumprod()

plt.plot(sim_cum_return.index,sim_cum_return,color='green')
plt.plot(test_cum_return.index,test_cum_return)
plt.show()