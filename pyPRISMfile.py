import pyPRISM
import numpy as np
import holoviews as hv
hv.extension('bokeh')
import matplotlib.pyplot as plt
import pandas as pd
def interpolate_guess(domain_from,domain_to,rank,guess):
    '''Helper for upscaling the intial guesses'''
    guess = guess.reshape((domain_from.length,rank,rank))
    new_guess = np.zeros((domain_to.length,rank,rank))
    for i in range(rank):
        for j in range(rank):
            new_guess[:,i,j] = np.interp(domain_to.r,domain_from.r,guess[:,i,j])
    return new_guess.reshape((-1,))

# fix variable
alp=0.25     # alpha = decase length, fixed
# coi polymer là 1 chuỗi nối liền với nhau, với đường kính không đổi
d = 1.0 #polymer segment diameter
eta = 0.4 #total occupied volume fraction => eta % tube là nanoparticle

def result_programme(epsilon_po_pa,epsilon_pa_pa,D_target,phi,chain_length_polymer):
    gr_result_load =[]
    #Chạy thử
    sys = pyPRISM.System(['particle','polymer'],kT=1.0)
    sys.domain = pyPRISM.Domain(dr=0.1,length=1024)

    guess = np.zeros(sys.rank*sys.rank*sys.domain.length)
    # D: đường kính của hạt nano.
    # d: đường kính của hạt polymer
    for D in np.arange(D_target-1,D_target+0.5,0.5):
        print('==> Solving for nanoparticle diameter D=',D)

        sys.diameter['polymer'] = d
        sys.diameter['particle'] = D
        sys.density['polymer'] = (1-phi)*eta/sys.diameter.volume['polymer']
        sys.density['particle'] = phi*eta/sys.diameter.volume['particle']
        #print('--> rho=',sys.density['polymer'],sys.density['particle'])

        # mô hình polymer: coi tất cả các phần tử polymer cạnh nhau là chuyển động tự do (Freely Jointed Chain)
        # length=100 => trên chuỗi polymer thì: N=100 segments (100 hạt nối liền với nhau)
        # l=4/3*d= khoảng cách giữa 2 hạt polymer (vì các hạt polymer overlap nhau nên kc giữa 2 tâm <2* bán kính)
        sys.omega['polymer','polymer'] = pyPRISM.omega.FreelyJointedChain(length=100,l=4.0*d/3.0)
        sys.omega['polymer','particle'] = pyPRISM.omega.InterMolecular()
        sys.omega['particle','particle'] = pyPRISM.omega.SingleSite()


        # tương tác: potential interaction: HardSphere model : ở xa thì k tương tác
        # polymer với particle thì có tương tác với nhau : -epsilon*epx((r-0.5*(p+d))/sigma)
        # particle vs particle: HardShere interaction

        sys.potential['polymer','polymer'] = pyPRISM.potential.HardSphere()
        sys.potential['polymer','particle'] = pyPRISM.potential.Exponential(alpha=alp,epsilon=1.0)
        sys.potential['particle','particle'] = pyPRISM.potential.Exponential(alpha=alp,epsilon=0.2)

        sys.closure['polymer','polymer'] = pyPRISM.closure.PercusYevick()
        sys.closure['polymer','particle'] = pyPRISM.closure.PercusYevick()
        sys.closure['particle','particle'] = pyPRISM.closure.HyperNettedChain()

        PRISM = sys.createPRISM()

        result = PRISM.solve(guess)

        guess = np.copy(PRISM.x)

        #print('')

        last_guess=guess
    print('Done!')

    # Chạy thật
    sys = pyPRISM.System(['particle','polymer'],kT=1.0)
    sys.domain = pyPRISM.Domain(dr=0.075,length=2048)

    # fix D= (giá trị cuối của vòng lặp trước). Để thay đổi bán kính particle thì thay đổi max của vòng loop
    sys.diameter['polymer'] = d
    sys.diameter['particle'] = D
    sys.density['polymer'] = (1-phi)*eta/sys.diameter.volume['polymer']
    sys.density['particle'] = phi*eta/sys.diameter.volume['particle']
    print('--> rho=',sys.density['polymer'],sys.density['particle'])

    sys.omega['polymer','polymer'] = pyPRISM.omega.FreelyJointedChain(length=100,l=4.0*d/3.0)
    sys.omega['polymer','particle'] = pyPRISM.omega.NoIntra()
    sys.omega['particle','particle'] = pyPRISM.omega.SingleSite()

    sys.closure['polymer','polymer'] = pyPRISM.closure.PercusYevick()
    sys.closure['polymer','particle'] = pyPRISM.closure.PercusYevick()
    sys.closure['particle','particle'] = pyPRISM.closure.HyperNettedChain()

    guess = interpolate_guess(pyPRISM.Domain(dr=0.1,length=1024),sys.domain,sys.rank,last_guess)

    for alpha in [0.25]:
        #print('==> Solving for alpha=',alpha)
        sys.potential['polymer','polymer'] = pyPRISM.potential.HardSphere()
        sys.potential['polymer','particle'] = pyPRISM.potential.Exponential(alpha=alpha,epsilon=epsilon_po_pa)
        # turn on tương tác giữa particle và particle
        sys.potential['particle','particle'] = pyPRISM.potential.Exponential(alpha=alpha,epsilon=epsilon_pa_pa)

        PRISM = sys.createPRISM()
        result = PRISM.solve(guess)

        x = sys.domain.r
        # hàm phân bố giữa các hạt với nhau
        y = pyPRISM.calculate.pair_correlation(PRISM)['particle','particle']
        gr_result_load.append([epsilon_po_pa,epsilon_pa_pa,D_target,sys.density['particle'],phi,chain_length_polymer,x,y])
        #print(epsilon_po_pa,epsilon_pa_pa,D_target,sys.density['particle'],phi,chain_length_polymer)
    return gr_result_load

    print('Done!')

resul = result_programme(0.2,0.4,4,0.001,20)
