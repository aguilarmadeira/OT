
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def top(nelx, nely, volfrac, penal, rmin):
    # INITIALIZE
    x = np.ones((nely, nelx)) * volfrac
    loop = 0
    change = 1.0

    # START ITERATION
    while change > 0.01:
        loop += 1
        xold = x.copy()

        # FE-ANALYSIS
        U = FE(nelx, nely, x, penal)

        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        KE = lk()
        c = 0.0
        dc = np.zeros((nely, nelx))
        for ely in range(nely):
            for elx in range(nelx):
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                Ue = U[[2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3]]
                c += (x[ely, elx] ** penal) * (Ue.T @ KE @ Ue).item()
                dc[ely, elx] = -penal * (x[ely, elx] ** (penal - 1)) * (Ue.T @ KE @ Ue).item()

        # FILTERING OF SENSITIVITIES
        dc = check(nelx, nely, rmin, x, dc)

        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD
        x = OC(nelx, nely, x, volfrac, dc)

        # PRINT RESULTS
        change = np.max(np.abs(x - xold))
        print(f" It.: {loop:4d} Obj.: {c:10.4f} Vol.: {np.sum(x)/(nelx*nely):6.3f} ch.: {change:6.3f}")

        # PLOT DENSITIES
        plt.imshow(-x, cmap='gray')
        plt.axis('equal')
        plt.axis('off')
        plt.pause(1e-6)
    plt.show()

def OC(nelx, nely, x, volfrac, dc):
    l1 = 0
    l2 = 100000
    move = 0.2
    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l2 + l1)
        xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
        if np.sum(xnew) - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew

def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))
    for i in range(nelx):
        for j in range(nely):
            sum = 0.0
            for k in range(max(i - int(rmin), 0), min(i + int(rmin) + 1, nelx)):
                for l in range(max(j - int(rmin), 0), min(j + int(rmin) + 1, nely)):
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    sum += max(0, fac)
                    dcn[j, i] += max(0, fac) * x[l, k] * dc[l, k]
            dcn[j, i] /= (x[j, i] * sum)
    return dcn

def FE(nelx, nely, x, penal):
    KE = lk()
    K = sp.lil_matrix((2 * (nelx + 1) * (nely + 1), 2 * (nelx + 1) * (nely + 1)))
    F = sp.lil_matrix((2 * (nely + 1) * (nelx + 1), 1))
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 1))

    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 2 * n1 + 2, 2 * n1 + 3])
            K[np.ix_(edof, edof)] += x[ely, elx] ** penal * KE

    # Convert K to CSR format
    K = K.tocsr()

    # DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
    F[1, 0] = -1
    fixeddofs = np.union1d(np.arange(0, 2 * (nely + 1), 2), np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    alldofs = np.arange(2 * (nely + 1) * (nelx + 1))
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    # SOLVING
    U[freedofs, 0] = spla.spsolve(K[freedofs][:, freedofs], F[freedofs].toarray().flatten())
    U[fixeddofs, 0] = 0
    return U

def lk():
    E = 1.0
    nu = 0.3
    k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
                  -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    KE = E / (1 - nu**2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                     [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                     [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                     [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                     [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                     [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                     [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                     [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE

# Exemplo de uso
top(60, 20, 0.5, 3, 1.5)