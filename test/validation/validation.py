from __future__ import print_function, division
from scipy.spatial import KDTree
import numpy as np
import matplotlib as mpl
if __name__=='__main__':
    mpl.use('Agg')
import matplotlib.pylab as plt
import trainio
import model
import fitsio
from CorrelationFunction import projected_correlation


def plotWpRp(ipos, ppos, outbase, zmax=40., h=0.7, Lb=400, rmin=0.1, rmax=2, rstep=0.1, ax = None ):
    """
    Make a comparison plot between the input projected correlation function
    and the predicted projected correlation function
    """
    Lb = Lb / h
    nrbins = ( rmax - rmin ) / rstep
    rbins = np.logspace( rmin, rmax, nrbins )
    rcen = ( rbins[:-1] + rbins[1:] ) / 2

    iwprp, icov = projected_correlation( ipos, rbins, zmax, Lb, jackknife_nside=3 )
    pwprp, pcov = projected_correlation( ppos, rbins, zmax, Lb, jackknife_nside=3 )
    iwpe = np.sqrt( np.diagonal( icov ) )
    pwpe = np.sqrt( np.diagonal( pcov ) )

    if ax == None:
        f, ax = plt.subplots(1)
    
    ax.set_yscale( 'log', nonposy = 'clip' )
    ax.set_xscale( 'log', nonposx = 'clip' )
    ax.set_ylabel( r'$w_{p}(r_{p})$', fontsize=20 )
    ax.set_xlabel( r'$r_{p} [Mpc\cdot h^{-1}]$', fontsize=20 )

    ax.errorbar( rcen, iwprp, yerr = iwpe, label='Original Halos' )
    ax.errorbar( rcen, pwprp, yerr = pwpe, label='Added Halos' )

    plt.legend()
    #plt.tight_layout()
    plt.savefig( outbase+'_wprp.png' )
    
    wdtype = np.dtype( [ ('r', float), ('iwprp', float), ('pwprp', float), 
                         ('iwprpe', float), ('pwprpe', float ) ] )
    wprp = np.ndarray( len( rcen ), dtype = wdtype )
    wprp[ 'r' ] = rcen
    wprp[ 'iwprp' ] = iwprp
    wprp[ 'pwprp' ] = pwprp
    wprp[ 'iwprpe' ] = iwpe
    wprp[ 'pwprpe' ] = pwpe

    fitsio.write(outbase+'_wprp.fit', wprp)

    return f, ax
    

def plotMassFunction(im, pm, outbase, mmin=9, mmax=13, mstep=0.05):
    """
    Make a comparison plot between the input mass function and the 
    predicted projected correlation function
    """
    plt.clf()

    nmbins = ( mmax - mmin ) / mstep
    mbins = np.logspace( mmin, mmax, nmbins )
    mcen = ( mbins[:-1] + mbins[1:] ) /2
    
    plt.xscale( 'log', nonposx = 'clip' )
    plt.yscale( 'log', nonposy = 'clip' )
    
    ic, e, p = plt.hist( im, mbins, label='Original Halos', alpha=0.5, normed = True)
    pc, e, p = plt.hist( pm, mbins, label='Added Halos', alpha=0.5, normed = True)
    
    plt.legend()
    plt.xlabel( r'$M_{vir}$' )
    plt.ylabel( r'$\frac{dN}{dM}$' )
    #plt.tight_layout()
    plt.savefig( outbase+'_mfcn.png' )
    
    mdtype = np.dtype( [ ('mcen', float), ('imcounts', float), ('pmcounts', float) ] )
    mf = np.ndarray( len(mcen), dtype = mdtype )
    mf[ 'mcen' ] = mcen
    mf[ 'imcounts' ] = ic
    mf[ 'pmcounts' ] = pc

    fitsio.write( outbase+'_mfcn.fit', mf )

def plotMassFunctionRatio(im, pm, outbase, mmin=9, mmax=13, mstep=0.05):
    """
    Make a comparison plot between the input mass function and the 
    predicted projected correlation function
    """
    gs = mpl.gridspec.GridSpec(4,1)
    f = plt.figure()

    nmbins = ( mmax - mmin ) / mstep
    mbins = np.logspace( mmin, mmax, nmbins )
    mcen = ( mbins[:-1] + mbins[1:] ) /2
    
    ic, e = np.histogram( im, bins=mbins )
    pc, e = np.histogram( pm, bins=mbins )
    rc = pc / ic

    ice = np.sqrt( ic )
    pce = np.sqrt( pc )
    rce = rc * np.sqrt( ( ice / ic ) ** 2 + ( pce / pc ) ** 2 )

    imsk = ( ice != 0 ) & ( ice == ice )
    pmsk = ( pce != 0 ) & ( pce == pce )
    rmsk = ( rce != 0 ) & ( rce == rce )
    
    ax = plt.subplot( gs[:2, :] )
    ax.set_xscale( 'log' )
    ax.set_yscale( 'log' )
    ax.set_ylabel( r'$\frac{dN}{dM}$' )

    ax.errorbar( mcen[ imsk ], ic[ imsk ], yerr = ice[ imsk ], label = 'Original Halos' )
    ax.errorbar( mcen[ pmsk ] , pc[ pmsk ], yerr = pce[ pmsk ], label = 'Added Halos' )

    plt.legend()

    ax = plt.subplot( gs[2:, :] )
    ax.set_xscale( 'log' )
    ax.set_yscale( 'log' )

    ax.errorbar( mcen[ rmsk ], rc[ rmsk ] , yerr = rce[ rmsk ] )
    ax.plot( mcen, np.zeros( len( mcen ) ) + 1, 'k' )
    ax.set_ylabel( r'$f_\frac{dN}{dM}$' )
    #ax = plt.subplot( gs[ :, : ] )
    ax.set_xlabel( r'$M_{vir}$' )

    plt.tight_layout()
    plt.savefig( outbase+'_mfcnr.png' )
    

def plotFeaturePDF(ift, pft, outbase, fmin=0.0, fmax=1.0, fstep=0.01):
    """
    Plot a comparison between the input feature distribution and the 
    feature distribution of the predicted halos
    """
    plt.clf()
    nfbins = ( fmax - fmin ) / fstep
    fbins = np.logspace( fmin, fmax, nfbins )
    fcen = ( fbins[:-1] + fbins[1:] ) / 2

    plt.xscale( 'log', nonposx='clip' )
    plt.yscale( 'log', nonposy='clip' )
    
    ic, e, p = plt.hist( ift, fbins, label='Original Halos', alpha=0.5, normed=True )
    pc, e, p = plt.hist( pft, fbins, label='Added Halos', alpha=0.5, normed=True )

    plt.legend()
    plt.xlabel( r'$\delta$' )
    plt.savefig( outbase+'_fpdf.png' )

    fdtype = np.dtype( [ ('fcen', float), ('ifcounts', float), ('pfcounts', float) ] )
    fd = np.ndarray( len(fcen), dtype = fdtype )
    fd[ 'mcen' ] = fcen
    fd[ 'imcounts' ] = ic
    fd[ 'pmcounts' ] = pc

    fitsio.write( outbase+'_fpdf.fit', fd )


def plotSSMassFunction(im, pm, ipos, ppos, outbase, mmin=9, mmax=13, mstep=0.05, side=2, boxsize=400):
    plt.clf()

    sidecuts = np.array( [ boxsize / ( 2 * side ) + boxsize * i / side for i in range( side ) ] )
    grid = np.meshgrid( sidecuts, sidecuts, sidecuts )
    garr = np.ndarray( ( side**3, 3 ) )
    garr[:,0] = grid[0].flatten()
    garr[:,1] = grid[1].flatten()
    garr[:,2] = grid[2].flatten()

    print(garr)

    tree = KDTree(garr)

    od, oii = tree.query( ipos )
    pd, pii = tree.query( ppos )

    print('oii: {0}'.format( np.unique( oii ) ))
    print('pii: {0}'.format( np.unique( pii ) ))

    nmbins = ( mmax - mmin ) / mstep
    mbins = np.logspace( mmin, mmax, nmbins )
    mcen = ( mbins[:-1] + mbins[1:] ) /2

    f, ax = plt.subplots( side**2, sharex=True, sharey=True)
    mjax = f.add_subplot(111)
    mjax.set_xlabel( r'$M_{vir}$' )
    mjax.set_ylabel( r'$\frac{dN}{dM}$' )

    tic, e = np.histogram( im, bins = mbins )
    tpc, e = np.histogram( pm, bins = mbins )
    
    for i in range( side**2 ):

        imask = ( oii == i )
        pmask = ( pii == i )

        ax[i].set_xscale( 'log', nonposx = 'clip' )
        ax[i].set_yscale( 'log', nonposy = 'clip' )

        sic, e = np.histogram( im[ imask ], bins=mbins )
        spc, e = np.histogram( pm[ pmask ], bins=mbins )
        
        fic = sic * side ** 3 / tic
        fpc = spc * side ** 3 / tpc

        fice = fic * np.sqrt( sic / sic ** 2 + tic / tic ** 2 )
        fpce = fpc * np.sqrt( spc / spc ** 2 + tpc / tpc ** 2 )
        ie = ( fice == fice ) & ( fice != 0 )
        pe = ( fpce == fpce ) & ( fpce != 0 )
        #ax[i].set_xlim([mbins[0]*.999, mbins[-1]*1.001])
        #ax[i].set_ylim([ np.min(np.hstack([fic,fpc]))*0.8, np.max(np.hstack([fic,fpc]))*1.2])
        
        ax[i].errorbar( mcen[ ie ], fic[ ie ] , yerr = fice[ ie ], label='Original Halos', \
                            markerfacecolor = 'None' )
        ax[i].errorbar( mcen[ pe ], fpc[ pe ] , yerr = fice[ pe ], label='Added Halos', \
                            markerfacecolor = 'None' )

    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)
    #plt.tight_layout()
    
    plt.savefig( outbase+'_ssmfcn.png' )
    

if __name__ == '__main__':

    #predpath = '/nfs/slac/g/ki/ki21/cosmo/jderose/halos/rockstar/output/FLb400/hlists/hlist_99.list'
    predpath = '/nfs/slac/g/ki/ki21/cosmo/jderose/halos/rockstar/output/FLb400/hlists/hlist_10.list'
    #hfeatpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir099/rnn_hlist_99'
    hfeatpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir010/rnn_hlist_10'
    hfeatdata = {hfeatpath:['hdelta']}
    preddata = {predpath:['x', 'y', 'z', 'M200b']}
    hfeatures = trainio.readData(hfeatdata)
    pred = trainio.readData(preddata)    

    pfeatpath = '/nfs/slac/g/ki/ki22/cosmo/jderose/halos/calcrnn/output/FLb400/snapdir010/parts/*rnn*snapshot_downsample_010.*'
    pfeatdata = {pfeatpath:['pdelta']}
    features = trainio.readData(pfeatdata)
    features = np.atleast_2d(features).T    

    ii = np.where( ( pred['M200b'] != 0 ) & ( pred['M200b'] >= 1e10 ) )
    hfeatures = hfeatures[ii]
    pred = pred[ii] 

    halos2 = fitsio.read('/nfs/slac/g/ki/ki22/cosmo/jderose/addhalos/halocats/FLb400/010_rfr/out0.list', ext=1)

    ohp = pred[['x', 'y', 'z']].view((pred['x'].dtype, 3))
    php = halos2[['PX', 'PY', 'PZ']].view((halos2['PX'].dtype, 3))
    im = pred['M200b']
    pm = halos2['M200b']
    outbase = 'plots/f400_rf_ufine_10'

    mcuts = [ 1e10, 1e11, 1e12 ]
    
    plotSSMassFunction(im, pm, ohp, php, outbase)
    plotMassFunctionRatio(im, pm, outbase)

    for i, mc in enumerate( mcuts ):
        omsk = ( im >= mc )
        pmsk = ( pm >= mc )
        if i == 0:
            f, ax = plotWpRp( ohp[ omsk ], php[ pmsk ], outbase + '_{0}'.format( mc ) )
        else:
            f, ax = plotWpRp( ohp[ omsk ], php[ pmsk ], outbase + '_{0}'.format( mc ), ax = ax)

    plotMassFunction(im, pm, outbase)
    plotFeaturePDF(hfeatures['hdelta'], halos2['pdelta'], outbase)
    

    
    

    
