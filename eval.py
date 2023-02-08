import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import h5py
import torch
import numpy as np


def fn(y):
  if y==0:
    return "p"
  elif y==1:
    return "ux"
  else:
    return "uy"

def loss_plot(loss_list, save_path, file):
  plt.ion()
  fig = plt.figure()
  plt.plot(loss_list)
  plt.xlabel('Batches')
  plt.ylabel('Loss')
  plt.savefig(save_path)

def show_image(img):
    npimg = img.numpy()
    plt.imshow(img.permute(1,2,0))

def plot_from_model(model_path, kle, datapoints, file):
  vae=torch.load(model_path)
  f2 = h5py.File("./dataset/kle" + str(kle) + "_mc500.hdf5", 'r')
  test=torch.Tensor(f2['input'])
  test_out=torch.Tensor(f2['output'])
  fig,ax=plt.subplots(1,1)
  maxi=torch.max(test[0][0]).detach().numpy()
  levels=np.linspace(0,maxi,10)
  np.round_(levels,decimals=2)
  cp = ax.contourf(test[0][0],levels=levels)
  fig.colorbar(cp)
  ax.set_ylabel('Y')
  ax.set_xlabel('X')
  plt.savefig(str(kle) + "_" + str(datapoints) + '_K.eps', format='eps')

  for i in range(0,3): 
    with torch.no_grad():
      images_recon= vae(test[0:2])
    fig,ax=plt.subplots(1,1)
    maxi1=max(torch.max(test_out[0][i]).detach().numpy(),torch.max(images_recon[0][i]).detach().numpy())
    min1=min(torch.min(test_out[0][i]).detach().numpy(),torch.min(images_recon[0][i]).detach().numpy())
    levels1=np.linspace(min1,maxi1,10)
    np.round(levels1,decimals=2)
    cp = ax.contourf(test_out[0][i],levels=levels1)
    fig.colorbar(cp)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.savefig('4225_128_'+fn(i)+'_in.eps', format='eps')
    plt.show()
    fig,ax=plt.subplots(1,1)
    with torch.no_grad():
      images_recon= vae(test[0:2])
      cp = ax.contourf(images_recon[0][i].detach().numpy(),levels=levels1)
      fig.colorbar(cp)
      ax.set_ylabel('Y')
      ax.set_xlabel('X')
      plt.savefig(str(kle) + "_" + str(datapoints) + fn(i)+'_out.eps', format='eps')
      plt.show()

def draw_pdf(model_path, kle, file):
  with torch.no_grad():
   vae=torch.load(model_path)
   f2 = h5py.File("./dataset/kle" + str(kle) + "_mc500.hdf5", 'r')
   t=torch.Tensor(f2['input'])
   t_out=torch.Tensor(f2['output'])
   mu,mu_test,sigma_test=vae.encoder(t)
   mu_final=vae.decoder(mu,mu_test)
   m=random.randint(0,64)
   n=random.randint(0,64)
   file.write(m)
   file.write(n)
   a=torch.zeros(size=(10000,4,1000))
   for i in range(0,10000):
     for j in range(0,4):
       if sigma_test[i][j]>0:
               a[i,j,:]=torch.from_numpy(np.random.normal(loc=mu_test[i][j], scale =torch.sqrt(sigma_test[i][j]),size=1000))
       else: 
             a[i,j,:]=mu_test[i][j]
   samples=torch.zeros(size=(10000,1,3))
   for i in range(0,100):
     sigma_final=vae.decoder(mu,a[:,:,i])[:,:,m,n].reshape(10000,1,3)
     samples=torch.cat((samples,sigma_final),dim=1)
     print(samples.shape)
   samples=samples.reshape(samples.shape[1],10000,3)
   for j in range(0,3):
    plt.hist(mu_final[:,j,m,n],50, density=True)
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    plt.show()
    x=np.linspace(mn,mx,1101)
    y_new=torch.zeros(1101,samples.shape[1])
    for i in range(0,samples.shape[1]):
      kde2=st.gaussian_kde(samples[1:,i,j])
      y_new[:,i]=torch.tensor(kde2.pdf(x))
    y_max=torch.max(y_new,dim=1)
    y_min=torch.min(y_new,dim=1)
    y_mean=torch.mean(y_new,dim=1)
    fig,ax=plt.subplots()
    ax.plot(x,y_max.values,color="red",linestyle="dashed",label="Confidence interval")
    ax.plot(x,y_min.values,color="red",linestyle="dashed")
    ax.fill_between(x,y_min.values,y_max.values,color='red',alpha=.1)
    ax.plot(x,y_mean,color="darkred",label="Predicted PDF Plot")
    khj=st.gaussian_kde(t_out[:,j,m,n])
    ax.plot(x,khj.pdf(x),color="blue",label="Actual PDF Plot")
    ax.set_ylabel('PDF')
    ax.set_xlabel('Pixel Value')
    ax.legend()
    plt.show()