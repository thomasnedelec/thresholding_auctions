from scipy import optimize
import numpy as np

def compute_reserve_price(virtual_value,bound):
    #if bound[1] < 0:
    #    print('biu')
    #    return 0
    reserve_price = optimize.brentq(virtual_value, bound[0], bound[1])
    print("fullfiled")
    return reserve_price

def virtual_value_realisation(x, cdf, pdf):
    return x - (1 - cdf(x))/pdf(x)

def virtual_value_function(distrib):
    return lambda x : x - (1 - distrib.cdf(x))/distrib.pdf(x)

def virtual_value_realisation_when_log_data(x, cdf, pdf):
    return 10**x - (1-cdf(x))*10**x*np.log(10) / pdf(x)

def virtual_value_function_when_log_data(distrib):
    return lambda x : 10**x - (1-distrib.cdf(x))*10**x*np.log(10) / distrib.pdf(x)


def virtual_value_function_when_log_data2(distrib):
    return lambda x : x - x*np.log(10)*(1-distrib.cdf(np.log10(x))) / distrib.pdf(np.log10(x))

def virtual_value_realisation_thresholded(x,cdf,pdf,reserve_value, epsilon):
    if 10**x >= reserve_value:
        return x - (1 - cdf(x)) / pdf(x)
    else:
        return epsilon*x
