import itertools
import experiment_loader
import ALA3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mpl
import scipy.stats  #Note no longe resi 0 but resi 1
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

bayesian_bootstrap_run_list = [0,1]

grid = itertools.product(ALA3.ff_list, ALA3.prior_list)
grid = [("charmm27", "maxent")]

for k, (ff, prior) in enumerate(grid):
    print(ff, prior)
    regularization_strength = ALA3.regularization_strength_dict[prior][ff]

    predictions, measurements, uncertainties = experiment_loader.load(ff)
    phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)
    p = np.vstack([np.loadtxt(ALA3.data_directory + "/frame_populations/pops_%s_%s_reg-%.1f-BB%d.dat" % (ff, prior, regularization_strength, bayesian_bootstrap_run)) for bayesian_bootstrap_run in bayesian_bootstrap_run_list]).mean(0)

    line_color = 'w'
    linewidth = 10
    prop = matplotlib.font_manager.FontProperties(size=8.0)

    vmax = 0.007
    p0 = p * 0.0 + 1. / float(len(p))
    h,x,y = np.histogram2d(phi,psi,bins=100,weights=p0)
    extent = [-180,180,-180,180]
    plt.imshow(h.T,origin='lower',extent=extent, vmin=0, vmax=vmax)

    plt.legend(loc=0,labelspacing=0.075,prop=prop,scatterpoints=1,markerscale=0.5,numpoints=1)
    plt.plot([-180,0],[50,50],line_color,linewidth=linewidth)
    plt.plot([-180,0],[-100,-100],line_color,linewidth=linewidth)
    plt.plot([0,180], [100,100],line_color,linewidth=linewidth)
    plt.plot([0,180], [-50,-50],line_color,linewidth=linewidth)
    plt.plot([0,0],[-180,180],line_color,linewidth=linewidth)
    plt.plot([-100,-100],[50,180],line_color,linewidth=linewidth)
    plt.plot([-100,-100],[-180,-100],line_color,linewidth=linewidth)

    plt.title("%s Raw" % ALA3.ff_map[ff])
    plt.xlabel(r"$\phi$ [$\circ$]")
    plt.ylabel(r"$\psi$ [$\circ$]")

    plt.annotate("PPII",[-60,150],color=line_color,fontsize='x-large')
    plt.annotate(r"$\beta$",[-130,150],color=line_color,fontsize='x-large')
    plt.annotate(r"$\alpha$",[-100,0],color=line_color,fontsize='x-large')
    plt.annotate(r"$\alpha_L$",[100,30],color=line_color,fontsize='x-large')
    plt.annotate(r"$\gamma$",[100,-150],color=line_color,fontsize='x-large')

    plt.axis([-180,180,-180,180])
    #plt.colorbar()
    plt.savefig(ALA3.outdir+"/ALA3_rama_%s_raw.pdf" % ff, bbox_inches='tight')

    h,x,y = np.histogram2d(phi,psi,bins=100,weights=p)
    plt.figure()
    plt.imshow(h.T,origin='lower',extent=extent, vmin=0, vmax=vmax)

    plt.legend(loc=0,labelspacing=0.075,prop=prop,scatterpoints=1,markerscale=0.5,numpoints=1)
    plt.plot([-180,0],[50,50],line_color,linewidth=linewidth)
    plt.plot([-180,0],[-100,-100],line_color,linewidth=linewidth)
    plt.plot([0,180], [100,100],line_color,linewidth=linewidth)
    plt.plot([0,180], [-50,-50],line_color,linewidth=linewidth)
    plt.plot([0,0],[-180,180],line_color,linewidth=linewidth)
    plt.plot([-100,-100],[50,180],line_color,linewidth=linewidth)
    plt.plot([-100,-100],[-180,-100],line_color,linewidth=linewidth)

    plt.title("%s %s" % (ALA3.ff_map[ff], prior))
    plt.xlabel(r"$\phi$ [$\circ$]")
    plt.ylabel(r"$\psi$ [$\circ$]")

    plt.annotate("PPII",[-60,150],color=line_color,fontsize='x-large')
    plt.annotate(r"$\beta$",[-130,150],color=line_color,fontsize='x-large')
    plt.annotate(r"$\alpha$",[-100,0],color=line_color,fontsize='x-large')
    plt.annotate(r"$\alpha_L$",[100,30],color=line_color,fontsize='x-large')
    plt.annotate(r"$\gamma$",[100,-150],color=line_color,fontsize='x-large')

    plt.axis([-180,180,-180,180])
    #plt.colorbar()
    plt.savefig(ALA3.outdir+"/ALA3_rama_%s_%s_belt.pdf" % (ff, prior), bbox_inches='tight')

    fig = pyplot.figure(figsize=(6,1.5))
    ax1 = fig.add_axes([0.1, 0.75, 0.8, 0.2])

    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal', ticks=[0.0, vmax / 2., vmax])
    cb1.set_label("Population")

    pyplot.show()

    plt.savefig(ALA3.outdir+"/ALA3_rama_colorbar.pdf", bbox_inches='tight')
    
