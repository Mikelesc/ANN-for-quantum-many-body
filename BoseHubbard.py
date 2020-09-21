import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import qutip
import time
# import netket as nk
from mpi4py import MPI
from scipy.sparse.linalg import eigsh

# Number of sites (Ns) and the local dimension of Hilbert space at each site (Nl)
Ns = 4
Nl = 3

#Operadtors from qutip
a_dag = qutip.operators.create(Nl)
a = qutip.operators.destroy(Nl)
n = qutip.operators.num(Nl)
a0 = np.kron(a, np.identity(Nl))
adag0 = np.kron(a_dag, np.identity(Nl))
adag1 = np.kron(np.identity(Nl), a_dag)
a1 = np.kron(np.identity(Nl), a)

#BOSE HUBBARD PARAMETERS
#Chemical potential
mu = 14.0
#Hopping
J = 1.0
#Hubbard interaction
U = 25.0

L = Ns

# CUSTOM
#1D lattice
edges = []
for i in range(L-1):
    edges.append([i, (i + 1) % L])
g = nk.graph.CustomGraph(edges)

#Ahora montamos el hamiltoniano en cada sitio
operators2 = []
sites = []

U2 = U*0.5
# Potencial local y de interaccion si hay más de uno (SUMA(Vi*ni + U*ni*(ni-1)))
for i in range(L):
    operators2.append((((-mu*n).full())).tolist())
    sites.append([i])

for i in range(L):
    operators2.append((((U2*(n*(n-1))).full())).tolist())
    sites.append([i])

# # # Movimiento (-J*SUMA(ai*a_dagj))
for i in range(L):
    operators2.append((-J*(np.dot(a0, adag1) + np.dot(adag0, a1))).tolist())
    sites.append([i, (i + 1) % L])
    
# Boson Hilbert Space
hi = nk.hilbert.Boson(graph=g, n_max=Nl-1)

# Our custom Bose-Hubbard hamiltonian
BH = nk.operator.LocalOperator(hi, operators=operators2, acting_on=sites)

# We declare the machine we want to use (our ansatz)
ma = nk.machine.RbmMultiVal(alpha=4, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Exact Sampler (ideally change this for faster runs)
sa = nk.sampler.ExactSampler(machine=ma)
# sa = nk.sampler.MetropolisExchange(machine=ma)
# sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)

# Declare the optimizer we want to use
op = nk.optimizer.AdaMax()

# Variational Monte Carlo 
vmc = nk.variational.Vmc(
    hamiltonian=BH,
    sampler=sa,
    optimizer=op,
    n_samples=10,
    diag_shift=0.1,
    use_iterative=True,
    method='Sr')

output = 'BH'+str(Ns)+str(Nl)+'size'+str(BH.to_sparse().size)

print("Starting RBM...")
start = time.time()

vmc.run(output_prefix=output, n_iter=750)

end = time.time()
print('RBM runtime in seconds: ', end - start)

#Lanczos
print("Starting LANCZOS.")
start = time.time()
res = nk.exact.lanczos_ed(BH, first_n=1, compute_eigenvectors=True)
print(res.eigenvalues)
end = time.time()
print('Lanczos runtime in seconds: ', end - start)