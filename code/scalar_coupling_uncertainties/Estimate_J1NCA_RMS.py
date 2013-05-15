import schwalbe_couplings
import numpy as np

x,y = np.loadtxt("./J1NCA_Schwalbe.csv",delimiter=",",skiprows=1).T

yhat = schwalbe_couplings.J1_N_CA(x*np.pi/180.)

err = (y - yhat)

rms = (err**2).mean()**0.5

xgrid = np.linspace(-180,180,300)
yhat = schwalbe_couplings.J1_N_CA(xgrid*np.pi/180.)
plot(xgrid,yhat)
plot(x,y,'o')



x,y = np.loadtxt("./J2NCA.csv",delimiter=",",skiprows=1).T
err = (x - y)
rms = (err**2).mean()**0.5
