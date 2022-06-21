import autograd.numpy as np
from autograd import grad

def Inverse(M):
    a,b = M.shape
    if(a!=b):
        raise ValueError("Only square matrices are invertible")
    i = np.eye(a,a)
    return np.linalg.lstsq(M,i)[0]

def main():
    A = np.matrix([[2.,0.,0.],[0.,2.,0.],[0.,0.,2.]])

    def objective(b):
        #x = np.linalg.solve(A,b)
        x = Inverse(A).dot(b)
        return np.sqrt(x[0])

    objective_grad = grad(objective)

    initial_b = np.array([0.,1.,0.])
    print(objective_grad(initial_b))

if __name__ == '__main__':
    main()
