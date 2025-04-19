import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
import mpmath as mp
import matplotlib.pyplot as plt
def Gen_Data(n, p, true_beta):
    X = np.random.normal(loc = 0, scale = 1, size = (n, p))

    true_y = X.dot(true_beta)

    Y = true_y + np.random.normal(loc = 0, scale = 1, size = n)

    return X, Y, true_y

def Check_KKT(XA, XAc, y, bhA, lamda, n):
    print("Check Active")
    
    e1 = y - np.dot(XA,bhA)
    e2 = np.dot(XA.T, e1)
    print(e2/lamda)
    
    if XAc is not None:
        print("Check In Active")
        
        e1 = y - np.dot(XA,bhA)
        e2 = np.dot(XAc.T, e1)
        print(e2/lamda)
def Weighted_Lasso(X,Y,lamda):
  n,p = X.shape

  wj = [0.1,0.1,0.5,0.1,0.5]
  wj = np.array(wj)

  X_star = X / wj

  lasso = LassoLars(alpha = lamda/n,fit_intercept=False,max_iter=10000000)
  lasso.fit(X_star,Y)
  Beta = lasso.coef_
  return Beta, wj
def compute_quotient(numerator, denominator):
    if denominator == 0:
        return np.inf

    quotient = numerator / denominator

    if quotient <= 0:
        return np.inf

    return quotient
def compute_step_size(X, yz, lamda, b):
    
    lasso_model = Lasso(alpha=lamda/X.shape[0],fit_intercept=False,tol = 1e-10, max_iter= 1000000000)
    lasso_model.fit(X, yz)
    
    Bhz = lasso_model.coef_
    
    Az = Bhz != 0
    Acz = Bhz == 0
    XAz = X[:, Az]
    XAcz = X[:, Acz]
    BhAz = Bhz[Az]
    BhAz = BhAz.reshape(-1,1)
    
    psi = np.array([])
    psi = np.dot(np.dot(np.linalg.inv(np.dot(XAz.T,XAz)),XAz.T),b)
    shAz = np.array([])
    gammaAz = np.array([])

    e1 = yz - np.dot(XAz, BhAz)

    e2 = np.dot(XAcz.T, e1)
    shAz = e2/lamda
    gammaAz = (np.dot(XAcz.T, b) - np.dot(np.dot(XAcz.T, XAz), psi))
            
    BhAz = BhAz.flatten()
    psi = psi.flatten()
    shAz = shAz.flatten()
    gammaAz = gammaAz.flatten()
    min1 = np.inf
    min2 = np.inf

    for j in range(len(psi)):
        numerator = - BhAz[j]
        denominator = psi[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min1:
            min1 = quotient

    for j in range(len(gammaAz)):
        numerator = (np.sign(gammaAz[j]) - shAz[j])*lamda
        denominator = gammaAz[j]
        quotient = compute_quotient(numerator, denominator)
        if quotient < min2:
            min2 = quotient

    return min(min1, min2), Az, Bhz

def compute_zk(y, eta, zk, n):
    
    In = np.identity(n)
    
    b = np.dot(np.dot(In,eta), np.linalg.inv(np.dot(eta.T,np.dot(In,eta))))
    a = np.dot(In - np.dot(b,eta.T), y).reshape(-1,1)
    
    yz = a + b * zk
    return yz , b
    
def compute_solution_path(X, Y, threshold, eta, n, lamda):
    zk = -threshold
    
    list_zk = [zk]
    list_Bhz = []
    list_Az = []
    
    while zk < threshold:
        
        yz, b = compute_zk(Y, eta, zk, n)
        
        tzk, Az, Bhz = compute_step_size(X, yz, lamda, b)
        
        zk = zk + tzk + 0.0001
         
        if zk < threshold:
            list_zk.append(zk)
        else :
            list_zk.append(threshold)
            
        list_Bhz.append(Bhz)
        list_Az.append(Az)
        
    return list_zk, list_Bhz, list_Az

def parametric_lasso_SI():
    n = 100
    p = 5
    true_beta = [2,2,0,2,0]
    X, y, true_y = Gen_Data(n, p, true_beta)
    Sigma = np.identity(n)
    lamda = 0.1
    w = [0.1,0.1,0.5,0.1,0.5]
    w = np.array(w)
    X = X / w
    lasso = LassoLars(alpha = lamda/n,fit_intercept=False,max_iter=10000000)
    lasso.fit(X,y)
    B = lasso.coef_
    print("OC")
    print(B)
    A = B != 0
    Ac = B == 0
    Bh = B[A]
    wh = w[A]
    XA = X[:, A]
    XAc = X[:, Ac]
    feature_selected = np.random.choice(Bh)
    j_selected = np.where(Bh == feature_selected)
    print(np.where(A)[0])
    ej = np.zeros((len(Bh), 1))
    ej[j_selected] = 1
    wj = wh[j_selected][0]
    eta = np.dot(np.linalg.pinv(XA).T, ej)
    eta = eta.reshape(n,1)
    etaTy = np.dot(eta.T, y)[0] /wj
    tn_mu = np.dot(eta.T, true_y)[0]/wj
    sigma = np.sqrt(np.dot(eta.T, np.dot(Sigma, eta))[0][0])/wj

    list_zk, list_Bhz, list_Az = compute_solution_path(X, y, 10, eta, n, lamda)
    indices = [[i for i, val in enumerate(sub_A) if val] for sub_A in list_Az]
    print('list_Az: ')
    print(indices)
    print('list_Bhz: ')
    print(list_Bhz)
    interval = []
    for i in range(len(list_Az)):
        if np.array_equal(A, list_Az[i]):
            interval.append([list_zk[i], list_zk[i+1] - 1e-10])
    print("OC")
    print(interval)
    
    new_interval = []
    for each_interval in interval:
        if len(new_interval) == 0:
            new_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_interval[-1][1]
            if abs(sub) < 0.01:
                new_interval[-1][1] = each_interval[1]
            else:
                new_interval.append(each_interval)

    interval = new_interval/wj

    numerator = 0
    denominator = 0
    for each_interval in interval:
        left = each_interval[0]
        right = each_interval[1]

        denominator = denominator + mp.ncdf((right-tn_mu)/sigma) - mp.ncdf((left-tn_mu)/sigma)

        if etaTy >= right:
            numerator = numerator + mp.ncdf((right-tn_mu)/sigma) - mp.ncdf((left-tn_mu)/sigma)
        elif (etaTy >= left) and (etaTy < right):
            numerator = numerator + mp.ncdf((etaTy-tn_mu)/sigma) - mp.ncdf((left-tn_mu)/sigma)

    try:
        cdf = float(numerator / denominator)
        p_value = 2 * min(cdf, 1 - cdf)
    except (ZeroDivisionError, ValueError):
        return None
    return p_value

    
max_iteration = 1000
list_p_values = []

count_rejects = 0

for iter in range(max_iteration):
    p_value = parametric_lasso_SI()
    
    list_p_values.append(p_value)
    print(iter)
    if p_value <= 0.05:
        count_rejects += 1
        
list_p_values = np.array(list_p_values)

print("FPR: ", count_rejects / max_iteration)
plt.hist(list_p_values, edgecolor = 'black')
plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
plt.show()  
