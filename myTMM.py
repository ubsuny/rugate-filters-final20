import numpy as np

def TMM(polar, n_x, x, th_i, lam):
    '''
    Here we are using Transfer Matrix Method(TMM) to calculate the reflectance of the Rugate filter of our
    chosen refractive index profile. For the theoretical explanation, please see wiki page of this repository.
    The input parameters are defined below.
    
    polar = The polarization, either 's' or 'p'.
    n_x   = The refractive index profile. (It should start with the average refractive index of the filter.)
    x     = The coordinate values of the discretized width of the filter.
    th_i  = The incident angle on the first layer.
    lam   = The wavelenght of incident light in vacuum.
    '''
    
    # We want the first refrative index to be 1 because assuminng that the outer medium is air.
    n_list = np.zeros(len(n_x)+1)        
    n_list[0] = 1 
    for i in range(len(n_x)):
        n_list[i+1] = n_x[i]
    
    

    
    # Keep the list of width of each layer.
    d = np.zeros(x.size+1)
    d[0] = np.inf
    d[-1] = np.inf
    for i in range(1,len(x)):
        d[i] = x[i]-x[i-1]
    
    
    n_list = np.array(n_list)
    d_list = np.array(d, dtype=float)
    
    n_layers = n_list.size
    
    # Let's do some unit tests by checking the inputs.
    # 1. We are only doing 1-D simulation.
    # 2. We need layers and refractive indeces arrays of same size.
    
    if (n_x.ndim != 1) or (x.ndim != 1) or (n_x.size != x.size):
        raise ValueError("Problem with n_x or x.")
    
    # 3. We also want to check that the incident angle is < pi/2 and >-pi/2 from the normal line.
    
    if (th_i>np.pi/2) or (th_i<-np.pi/2):
        raise ValueError("The incident angle is not in forward direction.")
    
        
    '''
    We are calculating the angles of refraction using incident angles to each layer.
    n_i*sin(th_i) = n_r*sin(th_r)
    '''
    th = np.arcsin(n_list[0]*np.sin(th_i) / n_list)
    
    kz_list = 2 * np.pi * n_list * np.cos(th) / lam
    
    # delta is the phase factor.
    delta = kz_list * d
    
    # Now depending on the polarization, we need reflection and transmission amplitude arrays.
    t_list = np.zeros(n_layers, dtype=complex)
    r_list = np.zeros(n_layers, dtype=complex)
    if polar == 's':
        for i in range(n_layers-1):
            r_list[i] = ((n_list[i] * np.cos(th[i]) - n_list[i+1] * np.cos(th[i+1])) /
                         (n_list[i] * np.cos(th[i]) + n_list[i+1] * np.cos(th[i+1])))
            t_list[i] = 2 * n_list[i] * np.cos(th[i]) / (n_list[i] * np.cos(th[i]) + n_list[i+1] * np.cos(th[i+1]))
    elif polar == 'p':
        for i in range(n_layers-1):
            r_list[i] = ((n_list[i+1] * np.cos(th[i]) - n_list[i] * np.cos(th[i+1])) /
                         (n_list[i+1] * np.cos(th[i]) + n_list[i] * np.cos(th[i+1])))
            t_list[i] = 2 * n_list[i] * np.cos(th[i]) / (n_list[i+1] * np.cos(th[i]) + n_list[i] * np.cos(th[i+1]))
    else:
        raise ValueError("Polarization must be 's' or 'p'")
    
    # Set up the matrix.
    M = np.zeros((n_layers, 2, 2), dtype=complex)
    
    for i in range(1, n_layers-1):
        M[i] = (1/t_list[i]) * np.dot(
            np.array(([[np.exp(-1j*delta[i]), 0], [0, np.exp(1j*delta[i])]]), dtype=complex),
            np.array(([[1, r_list[i]], [r_list[i], 1]]), dtype=complex))
        
    M_tilde = np.identity(2)
    for i in range(1, n_layers-1):
        M_tilde = np.dot(M_tilde, M[i])
    M_tilde = np.dot(np.array(([[1, r_list[0]], [r_list[0], 1]]), dtype=complex)/t_list[0], M_tilde)
    
    # Net transmission and reflection amplitudes.
    r = M_tilde[1,0]/M_tilde[0,0]
    t = 1/M_tilde[0,0]
    
    # Now get the reflectance.
    R = abs(r)**2
    
    if polar == 's':
        T = abs(t**2) * (((n_list[-1]*np.cos(th[-1])).real) / (n_list[0]*np.cos(th[0])).real)
    elif polar == 'p':
        T = abs(t**2) * (((n_list[-1]*np.conj(np.cos(th[-1]))).real) / (n_list[0]*np.conj(np.cos(th[0]))).real)
    else:
        raise ValueError("Polarization must be 's' or 'p'")
    
    return R,T