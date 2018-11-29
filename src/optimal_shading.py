
from scipy import optimize
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
from src.utils import *


def compute_optimal_shading(K, cdf, pdf, type_shading, setting,inf,sup):
        if setting == "symmetric":
            a, b, x_opt, vrt_val_xopt = equations_symmetric_case(cdf, pdf, type_shading, K, inf, sup)
        elif setting == "one_strategic":
            a, b, x_opt, vrt_val_xopt = equations_one_strategic_case(cdf, pdf, type_shading, K, inf, sup)
        else:
            print("Wrong arguments of compute_optimal_shading")
        print("beta(x) = a*x+b")
        print(f"a:{a}, b:{b}")
        print(f"x_opt:{x_opt}")
        print(f"vrt_val_xopt:{vrt_val_xopt}")
        return a,b


def equations_symmetric_case(cdf,pdf,type_shading,K,inf,sup):
    if type_shading == "affine":
        equation = lambda t: system_symmetric(t,cdf,pdf,K,sup)
        x_opt = optimize.brentq(equation, inf, 0.99*sup)
        vrt_val_xopt = virtual_value_realisation(x_opt, cdf, pdf)
    elif type_shading == "linear":
        equation = lambda t: virtual_value_realisation(t, cdf, pdf)
        x_opt = optimize.brentq(equation, inf, 0.99*sup)
        vrt_val_xopt = virtual_value_realisation(x_opt, cdf, pdf)
    equation = lambda a: J1(a, cdf, pdf, K, sup, x_opt, vrt_val_xopt)
    a_opt = optimize.brentq(equation, inf, sup)
    b_opt = -vrt_val_xopt*a_opt
    return a_opt, b_opt, x_opt, vrt_val_xopt


def equations_one_strategic_case(cdf, pdf, type_shading, K, inf, sup, xinit=[0.7, 0.2]):
    if type_shading == "linear":
        equation = lambda t: virtual_value_realisation(t, cdf, pdf)
        x_opt = optimize.brentq(equation, 0.001, 0.99*sup)
        vrt_val_xopt = virtual_value_realisation(x_opt, cdf, pdf)
        equation = lambda a : A2(a, cdf, pdf, K, sup, x_opt) - B2(a, cdf, pdf, K, sup, x_opt)
        a_opt = optimize.brentq(equation, 0.001, sup)
        b_opt = 0
    elif type_shading == "affine":
        xinit = xinit
        system = lambda z : system_one_strategic(z, cdf, pdf, K, sup)
        a_opt, x_opt = fsolve(system,xinit)
        vrt_val_xopt = virtual_value_realisation(x_opt, cdf, pdf)
        b_opt = -virtual_value_realisation(x_opt, cdf, pdf) * a_opt
    return a_opt, b_opt, x_opt, vrt_val_xopt



def indicate(x, cdf, pdf, t):
    if virtual_value_realisation(x,cdf,pdf) > t:
        return 1
    else:
        return 0

#equations symmetric
def L1(t, cdf, pdf, K, sup):
    y = -virtual_value_realisation(t, cdf, pdf)*(cdf(t)**(K-1)+integrate.quad(lambda z: (K-1)*pdf(z)*cdf(z)**(K-2), t, sup)[0])
    return y


def L2(t, cdf, pdf, K, sup):
    y = cdf(t)**(K-1)*t*(-virtual_value_realisation(t, cdf, pdf) + (1-cdf(t))) -virtual_value_realisation(t, cdf, pdf) *\
        integrate.quad(lambda z: (K-1)*z*pdf(z)*cdf(z)**(K-2), t, sup)[0]
    return y


def R1(t, cdf, pdf, K, sup):
    y = integrate.quad(lambda z: z*(K-1)*pdf(z)*cdf(z)**(K-2), t, sup)[0]
    return y


def R2(t, cdf, pdf, K, sup):
    y = integrate.quad(lambda z: z**2*(K-1)*pdf(z)*cdf(z)**(K-2), t, sup)[0]
    return y


def system_symmetric(t, cdf, pdf, K, sup):
    y = L1(t, cdf, pdf, K, sup)*R2(t, cdf, pdf, K, sup) - L2(t, cdf, pdf, K, sup)*R1(t, cdf, pdf, K, sup)
    return y


def J1(a, cdf, pdf, K, sup, x_opt, vrt_val_xopt):
    if K >1:
        y1 = cdf(x_opt)**(K-1)*x_opt*(-vrt_val_xopt+(1-cdf(x_opt)))
        y2 = -vrt_val_xopt * integrate.quad(lambda z: (K-1) *\
            pdf(z)*cdf(z)**(K-2)*z*indicate(z, cdf, pdf, vrt_val_xopt), 0, sup)[0]
        y3 = integrate.quad(lambda z: z**2*(K-1)*pdf(z)*\
            cdf(z)**(K-2)*indicate(z, cdf, pdf, vrt_val_xopt), 0, sup)[0]
        y = a*(y1+y2) - (1-a)*y3
    else:
        y = a - 0.01
    return y

# equations one strategic

def A1(a, cdf, pdf, K, sup, x_opt):
    b = -virtual_value_realisation(x_opt, cdf, pdf)*a
    y = integrate.quad(lambda z: (K-1) *\
            pdf(a*z+b)*cdf(a*z+b)**(K-2)*((1-a)*z - b), x_opt, sup)[0]
    return y


def B1(a, cdf, pdf, K, sup, x_opt):
    b = -virtual_value_realisation(x_opt, cdf, pdf) * a
    y = b/a*cdf(a*x_opt+b)**(K-1)*pdf(x_opt)
    return y


def A2(a, cdf, pdf, K, sup, x_opt):
    b = -virtual_value_realisation(x_opt, cdf, pdf) * a
    y = integrate.quad(lambda z: (K-1) *\
            pdf(a*z+b)*cdf(a*z+b)**(K-2)*((1-a)*z - b)*z, x_opt, sup)[0]
    return y


def B2(a, cdf, pdf, K, sup, x_opt):
    b = -virtual_value_realisation(x_opt, cdf, pdf) * a
    y = cdf(a*x_opt+b)**(K-1)*x_opt*(1-cdf(x_opt) + b/a * pdf(x_opt))
    return y


def system_one_strategic(tuple, cdf, pdf, K, sup):
    a, x_opt = tuple
    f1 = A1(a, cdf, pdf, K, sup, x_opt) - B1(a, cdf, pdf, K, sup, x_opt)
    f2 = A2(a, cdf, pdf, K, sup, x_opt) - B2(a, cdf, pdf, K, sup, x_opt)
    return f1, f2


# equations optimal threshold
def find_optimal_virtual_value_threshhold(cdf,pdf,K,inf,sup):
        equation = lambda t : equation_threshhold(t,cdf,pdf,K,inf,sup)
        r_opt = optimize.brentq(equation, inf+0.001, 0.99*sup)
        return r_opt


def equation_threshhold(t,cdf,pdf,K,inf,sup):
    y = integrate.quad(lambda z: virtual_value_realisation(z, cdf, pdf)*cdf(z)**(K-1), inf, t)[0]
    return y