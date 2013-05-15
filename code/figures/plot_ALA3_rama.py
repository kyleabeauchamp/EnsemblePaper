import matplotlib.pyplot as plt
import numpy as np
import experiment_loader
import matplotlib.font_manager
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import ALA3

ff = "charmm27"

directory = "/home/kyleb/dat/lvbp/%s/" % ff
keys, measurements, predictions, uncertainties, phi, psi = experiment_loader.load(directory)
model_directory = "/home/kyleb/dat/lvbp/%s/models-%s/" % (ff, ALA3.model)
p = np.loadtxt(model_directory + "/reg-%d-frame-populations.dat" % ALA3.regularization_strength_dict[ff])

line_color = 'w'
linewidth = 10
prop = matplotlib.font_manager.FontProperties(size=8.0)

h,x,y = np.histogram2d(phi,psi,bins=100)
extent = [-180,180,-180,180]
plt.imshow(h.T,origin='lower',extent=extent)

plt.legend(loc=0,labelspacing=0.075,prop=prop,scatterpoints=1,markerscale=0.5,numpoints=1)
plt.plot([-180,0],[50,50],line_color,linewidth=linewidth)
plt.plot([-180,0],[-100,-100],line_color,linewidth=linewidth)
plt.plot([0,180], [100,100],line_color,linewidth=linewidth)
plt.plot([0,180], [-50,-50],line_color,linewidth=linewidth)
plt.plot([0,0],[-180,180],line_color,linewidth=linewidth)
plt.plot([-100,-100],[50,180],line_color,linewidth=linewidth)
plt.plot([-100,-100],[-180,-100],line_color,linewidth=linewidth)

plt.title("%s Raw" % ff)
plt.xlabel(r"$\phi$ [$\circ$]")
plt.ylabel(r"$\psi$ [$\circ$]")

plt.annotate("PPII",[-60,150],color=line_color,fontsize='x-large')
plt.annotate(r"$\beta$",[-130,150],color=line_color,fontsize='x-large')
plt.annotate(r"$\alpha$",[-100,0],color=line_color,fontsize='x-large')
plt.annotate(r"$\alpha_L$",[100,30],color=line_color,fontsize='x-large')
plt.annotate(r"$\gamma$",[100,-150],color=line_color,fontsize='x-large')

plt.axis([-180,180,-180,180])
plt.savefig(ALA3.outdir+"/ALA3_rama_%s_raw.pdf" % ff, bbox_inches='tight')

h,x,y = np.histogram2d(phi,psi,bins=100,weights=p)
plt.figure()
plt.imshow(h.T,origin='lower',extent=extent)

plt.legend(loc=0,labelspacing=0.075,prop=prop,scatterpoints=1,markerscale=0.5,numpoints=1)
plt.plot([-180,0],[50,50],line_color,linewidth=linewidth)
plt.plot([-180,0],[-100,-100],line_color,linewidth=linewidth)
plt.plot([0,180], [100,100],line_color,linewidth=linewidth)
plt.plot([0,180], [-50,-50],line_color,linewidth=linewidth)
plt.plot([0,0],[-180,180],line_color,linewidth=linewidth)
plt.plot([-100,-100],[50,180],line_color,linewidth=linewidth)
plt.plot([-100,-100],[-180,-100],line_color,linewidth=linewidth)

plt.title("%s LVBP" % ff)
plt.xlabel(r"$\phi$ [$\circ$]")
plt.ylabel(r"$\psi$ [$\circ$]")

plt.annotate("PPII",[-60,150],color=line_color,fontsize='x-large')
plt.annotate(r"$\beta$",[-130,150],color=line_color,fontsize='x-large')
plt.annotate(r"$\alpha$",[-100,0],color=line_color,fontsize='x-large')
plt.annotate(r"$\alpha_L$",[100,30],color=line_color,fontsize='x-large')
plt.annotate(r"$\gamma$",[100,-150],color=line_color,fontsize='x-large')

plt.axis([-180,180,-180,180])
plt.savefig(ALA3.outdir+"/ALA3_rama_%s_lvbp.pdf" % ff, bbox_inches='tight')