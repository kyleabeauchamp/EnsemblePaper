import matplotlib.pyplot as plt
import numpy as np
import schwalbe_couplings
import experiment_loader
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

outdir = "/home/kyleb/src/pymd/Papers/maxent/figures/"

alpha = 0.2
num_grid = 500
phi = np.linspace(-180,180,num_grid)
O = np.ones(num_grid)
colors = ["b","g","r","y"]

simulation_data = {}

simulation_data["J3_HN_HA_2"] = schwalbe_couplings.J3_HN_HA(phi)
simulation_data["J3_HN_Cprime_2"] = schwalbe_couplings.J3_HN_Cprime(phi)
simulation_data["J3_HA_Cprime_2"] = schwalbe_couplings.J3_HA_Cprime(phi)
simulation_data["J3_HN_CB_2"] = schwalbe_couplings.J3_HN_CB(phi)
simulation_data["J1_N_CA_2"] = schwalbe_couplings.J1_N_CA(phi)
simulation_data["J2_N_CA_3"] = schwalbe_couplings.J2_N_CA(phi)

plt.plot(phi,simulation_data["J3_HN_HA_2"],'b',label="Karplus Curve")
yi = experiment_loader.experimental_data["J3_HN_HA_2"]
oi = experiment_loader.sigma_dict["J3_HN_HA"]
plt.plot(phi,O*yi,"k",label="Measured Average")
lower = yi - oi
upper = yi + oi
plt.fill_between(phi,lower*O,upper*O,color='k',alpha=alpha)
plt.xlabel(r"$\phi$ [$\circ$]")
plt.ylabel(r"$f_i(\phi) = $ J")
plt.title("Projecting $\phi$ onto J Couplings")
plt.legend(loc=0)
plt.xlim(-180,180)
plt.ylim(-0.5,10.5)
#plt.savefig(outdir+"/single_karplus.pdf",bbox_inches='tight')
plt.figure()



for i,key in enumerate(["J3_HN_HA_2","J3_HN_Cprime_2","J3_HA_Cprime_2","J3_HN_CB_2"]):
    plt.plot(phi,simulation_data[key],"%s" % colors[i])
    yi = experiment_loader.experimental_data[key]
    oi = experiment_loader.sigma_dict[key[:-2]]
    plt.plot(phi,O*yi,"%s" % colors[i])    
    plt.plot(phi,O*yi,"%s" % colors[i])
    lower = yi - oi
    upper = yi + oi
    plt.fill_between(phi,lower*O,upper*O,color=colors[i],alpha=alpha)
    

plt.xlabel(r"$\phi$ [$\circ$]")
plt.ylabel(r"$f_i(\phi) = $ J")
plt.xlim(-180,180)
plt.ylim(-0.5,10.5)
plt.title("Projecting $\phi$ onto J Couplings")    
plt.savefig(outdir+"/multiple_karplus.pdf",bbox_inches='tight')
