# here i take all walkers and do trace plots, corner plots and histograms
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, bernoulli, expon, uniform, beta, gamma, multinomial, multivariate_normal
from scipy.stats import rv_histogram
from scipy.special import digamma
import random
from scipy.special import gamma as gamma_function
from scipy.special import gammaln
from scipy.special import factorial
from scipy.special import beta as beta_function
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import dirichlet
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import corner

import necessary_functions as nf
from emcee import autocorr

data_dir = sys.argv[1]
nwalkers = int(sys.argv[2])
N=int(sys.argv[3])	
T=int(sys.argv[4])
burnout=int(sys.argv[5])
keep_every=int(sys.argv[6])
Nprior=10

data = np.loadtxt(data_dir+'/processed_data.dat')
data_smeared = np.loadtxt(data_dir+'/processed_data_smeared.dat')

labels=data[:,2]
f1=np.sum(labels==1)/len(labels)

ohe_nj=OneHotEncoder(handle_unknown='error')
ohe_nb=OneHotEncoder(handle_unknown='error')
Y1=ohe_nj.fit_transform(data[:,0].reshape(-1,1)).toarray()
Y2=ohe_nb.fit_transform(data[:,1].reshape(-1,1)).toarray()

X=[]
for n in range(Y1.shape[0]):
  X.append([Y1[n],Y2[n]])

true_alphas=np.zeros((2,Y1.shape[1]))
true_betas=np.zeros((2,Y2.shape[1]))
for k in range(2):
  true_alphas[k]=np.mean(Y1[labels==k],axis=0)
  true_betas[k]=np.mean(Y2[labels==k],axis=0)
  
Y1_smeared=ohe_nj.transform(data_smeared[:,0].reshape(-1,1)).toarray()
Y2_smeared=ohe_nb.transform(data_smeared[:,1].reshape(-1,1)).toarray()
fake_alphas=np.zeros((2,Y1.shape[1]))
fake_betas=np.zeros((2,Y2.shape[1]))
for k in range(2):
  fake_alphas[k]=np.mean(Y1_smeared[data_smeared[:,2]==k],axis=0)
  fake_betas[k]=np.mean(Y2_smeared[data_smeared[:,2]==k],axis=0)
  

K=true_alphas.shape[0]
dj=true_alphas.shape[1]
db=true_betas.shape[1]


Z_list=np.zeros((nwalkers,T,N,K))
pie_list=np.zeros((nwalkers,T,K))
alphas_list=np.zeros((nwalkers,T,K,dj))
betas_list=np.zeros((nwalkers,T,K,db))

pie_prior_list=np.zeros((nwalkers,T,K))
alphas_prior_list=np.zeros((nwalkers,T,K,dj))
betas_prior_list=np.zeros((nwalkers,T,K,db))

for walker in range(nwalkers):
  Z_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/Z_list.npy')
  pie_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/pie_list.npy')
  pie_prior_list[walker]=dirichlet.rvs(size=T,alpha=np.ones(2))
  alphas_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/alphas_list.npy')
  alphas_prior_list[walker,:,0,:]=dirichlet.rvs(size=T,alpha=Nprior*fake_alphas[0])
  alphas_prior_list[walker,:,1,:]=dirichlet.rvs(size=T,alpha=Nprior*fake_alphas[1])
  betas_list[walker]=np.load(data_dir+'/walker_'+str(walker+1)+'/betas_list.npy')
  betas_prior_list[walker,:,0,:]=dirichlet.rvs(size=T,alpha=Nprior*fake_betas[0])
  betas_prior_list[walker,:,1,:]=dirichlet.rvs(size=T,alpha=Nprior*fake_betas[0])

dim=K+K*dj+K*db

print("Prior")
print(r'$\pi_{1}$')
print(autocorr.integrated_time(pie_prior_list[:,:,1]))
for j in range(dj):
  print(r'$\alpha_{0'+str(j)+'}$')
  print(autocorr.integrated_time(alphas_prior_list[:,:,0,j]))
  print(r'$\alpha_{1'+str(j)+'}$')
  print(autocorr.integrated_time(alphas_prior_list[:,:,1,j]))
for b in range(db):
  print(r'$\beta_{0'+str(j)+'}$')
  print(autocorr.integrated_time(betas_prior_list[:,:,0,b]))
  print(r'$\beta_{1'+str(j)+'}$')
  print(autocorr.integrated_time(betas_prior_list[:,:,1,b]))

autocorrelations=np.zeros(dim)  
print("Posterior")
print(r'$\pi_{1}$')

autocorrelations[0]=autocorr.integrated_time(pie_list[:,:,0])
autocorrelations[1]=autocorr.integrated_time(pie_list[:,:,1])
print(autocorr.integrated_time(pie_list[:,:,1]))
for j in range(dj):
  print(r'$\alpha_{0'+str(j)+'}$')
  print(autocorr.integrated_time(alphas_list[:,:,0,j]))
  autocorrelations[K+j]=autocorr.integrated_time(alphas_list[:,:,0,j])
  print(r'$\alpha_{1'+str(j)+'}$')
  print(autocorr.integrated_time(alphas_list[:,:,1,j]))
  autocorrelations[K+dj+j]=autocorr.integrated_time(alphas_list[:,:,1,j])
for b in range(db):
  print(r'$\beta_{0'+str(j)+'}$')
  print(autocorr.integrated_time(betas_list[:,:,0,b]))
  autocorrelations[K+K*dj+b]=autocorr.integrated_time(betas_list[:,:,0,b])
  print(r'$\beta_{1'+str(j)+'}$')
  print(autocorr.integrated_time(betas_list[:,:,1,b]))
  autocorrelations[K+K*dj+db+b]=autocorr.integrated_time(betas_list[:,:,1,b])


README=open(data_dir +'/README.txt',"wt")
README.write('pi0: Prior %8.3f \t Posterior %8.3f \n'% (autocorr.integrated_time(pie_prior_list[:,:,1]),autocorr.integrated_time(pie_list[:,:,0])))
README.write('pi1: Prior %8.3f \t Posterior %8.3f \n'% (autocorr.integrated_time(pie_prior_list[:,:,1]),autocorr.integrated_time(pie_list[:,:,1])))
for j in range(dj):
  README.write('alpha0%d\t: Prior %8.3f \t Posterior %8.3f \n'% (j,autocorr.integrated_time(alphas_prior_list[:,:,0,j]),autocorr.integrated_time(alphas_list[:,:,0,j])))
  README.write('alpha1%d\t: Prior %8.3f \t Posterior %8.3f \n'% (j,autocorr.integrated_time(alphas_prior_list[:,:,1,j]),autocorr.integrated_time(alphas_list[:,:,1,j])))
for b in range(db):
  README.write('beta0%d\t: Prior %8.3f \t Posterior %8.3f \n'% (b,autocorr.integrated_time(betas_prior_list[:,:,0,b]),autocorr.integrated_time(betas_list[:,:,0,b])))
  README.write('beta1%d\t: Prior %8.3f \t Posterior %8.3f \n'% (b,autocorr.integrated_time(betas_prior_list[:,:,1,b]),autocorr.integrated_time(betas_list[:,:,1,b])))

README.close()


Nbis = np.exp(np.linspace(np.log(100), np.log(T), 10)).astype(int)
new = np.empty(len(Nbis))
for i, n in enumerate(Nbis):
    new[i] = autocorr.integrated_time(pie_list[:n,:,1])

# Plot the comparisons
plt.loglog(Nbis, new, "o-", label="new")
ylim = plt.gca().get_ylim()
plt.plot(Nbis, Nbis / 50.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates for $\pi_{1}$")
plt.legend(fontsize=14)
plt.savefig(data_dir+'/pi1_tau_estimates.pdf')
plt.savefig(data_dir+'/pi1_tau_estimates.png')


# here I thin the files automatically
tau_max=int(round(np.max(autocorrelations),0))


README=open(data_dir +'/README.txt',"a")
README.write('Thinned again with tau %d \n'% tau_max)
README.write('pi0: Posterior %8.3f \t Thinned Posterior %8.3f \n'% (autocorrelations[0],autocorr.integrated_time(nf.thin_a_sample(pie_list[:,:,0],tau_max))))
README.write('pi1: Posterior %8.3f \t Thinned Posterior %8.3f \n'% (autocorrelations[1],autocorr.integrated_time(nf.thin_a_sample(pie_list[:,:,1],tau_max))))
for j in range(dj):
  README.write('alpha0%d\t: Posterior %8.3f \t Thinned Posterior %8.3f \n'% (j,autocorrelations[K+j],autocorr.integrated_time(nf.thin_a_sample(alphas_list[:,:,0,j],tau_max))))
  README.write('alpha1%d\t: Posterior %8.3f \t Thinned Posterior %8.3f \n'% (j,autocorrelations[K+dj+j],autocorr.integrated_time(nf.thin_a_sample(alphas_list[:,:,1,j],tau_max))))
for b in range(db):
  README.write('beta0%d\t: Posterior %8.3f \t Thinned Posterior %8.3f \n'% (b,autocorrelations[K+K*dj+b],autocorr.integrated_time(nf.thin_a_sample(betas_list[:,:,0,b],tau_max))))
  README.write('beta1%d\t: Posterior %8.3f \t Thinned Posterior %8.3f \n'% (b,autocorrelations[K+K*dj+db+b],autocorr.integrated_time(nf.thin_a_sample(betas_list[:,:,1,b],tau_max))))

README.close()


np.save(data_dir+'/thinned_Z_list.npy',nf.thin_a_sample(Z_list,tau_max))
np.save(data_dir+'/thinned_pie_list.npy',nf.thin_a_sample(pie_list,tau_max))
np.save(data_dir+'/thinned_alphas_list.npy',nf.thin_a_sample(alphas_list,tau_max))
np.save(data_dir+'/thinned_betas_list.npy',nf.thin_a_sample(betas_list,tau_max))
