import numpy as np
import schwalbe_couplings
import matplotlib.pyplot as plt

data = np.loadtxt("/home/kyleb/src/pymd/Papers/maxent/code/ubiquitin.csv")


xgrid = np.linspace(-180,180,500)
xgrid_shaped = xgrid.reshape((1,-1,1))
all_phi_psi = data[:,-1].reshape((1,-1,1))
phi = all_phi_psi[0,:,0]

predicted = schwalbe_couplings.J3_HN_HA(all_phi_psi,0)
true = data[:,0]
plt.plot(true,predicted,'o')
err = predicted - true
(err**2).mean()**0.5



plt.figure()
predicted = schwalbe_couplings.J3_HA_Cprime(all_phi_psi,0)
true = data[:,1]
plt.plot(true,predicted,'o')
err = predicted - true
(err**2).mean()**0.5



plt.figure()
predicted = schwalbe_couplings.J3_HN_CB(all_phi_psi,0)
true = data[:,2]
plt.plot(true,predicted,'o')
err = predicted - true
(err**2).mean()**0.5


plt.figure()
predicted = schwalbe_couplings.J3_HN_Cprime(all_phi_psi,0)
true = data[:,3]
plt.plot(true,predicted,'o')
err = predicted - true
(err**2).mean()**0.5



true = data[:,0]
plt.errorbar(phi,true,yerr=0.36,fmt='o')
ygrid = schwalbe_couplings.J3_HN_HA(xgrid_shaped,0)
plt.plot(xgrid,ygrid,label="bax")
ygrid = schwalbe_couplings.J3_HN_HA_schwalbe(xgrid_shaped,0)
plt.plot(xgrid,ygrid,label="schwalbe")
ygrid = schwalbe_couplings.J3_HN_HA_ruterjans(xgrid_shaped,0)
plt.plot(xgrid,ygrid,label="ruterjans")


predicted = schwalbe_couplings.J3_HN_HA_bax(all_phi_psi,0)
err = predicted - true
(err**2).mean()**0.5

predicted = schwalbe_couplings.J3_HN_HA_schwalbe(all_phi_psi,0)
err = predicted - true
(err**2).mean()**0.5

predicted = schwalbe_couplings.J3_HN_HA_ruterjans(all_phi_psi,0)
err = predicted - true
(err**2).mean()**0.5
