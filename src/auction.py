import src.utils as utils
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np

class Bidder:
    def __init__(self, distrib, shading_factor= [1, 0]):
        self.distrib = distrib
        self.shading_factor = shading_factor
        self.current_value = 0
        self.bound = self.define_bound()
        self.revenue = 0

    def sample_value(self):
        r = self.distrib.rvs()
        self.current_value = r
        return r

    def bid(self):
        return self.shading_factor[0] * self.sample_value() + self.shading_factor[1]

    def formula_bid(self, x):
        return self.shading_factor[0] * x + self.shading_factor[1]

    def virtual_value(self, x):
        return self.formula_bid(x) - self.shading_factor[0] * (1 - self.distrib.cdf(x)) / (
                               self.distrib.pdf(x))

    def update_revenue(self, payment):
        self.revenue += self.current_value - payment

    def define_bound(self):
            inf = self.distrib.ppf(0.01)
            sup = self.distrib.ppf(0.99)
            return [inf,sup]

class Bidder_seuille:
    def __init__(self, distrib, seuille = 0.5):
        self.distrib = distrib
        self.current_value = 0
        self.bound = self.define_bound()
        self.revenue = 0
        self.epsilon = 0.001
        self.seuille = seuille

    def sample_value(self):
        r = self.distrib.rvs()
        self.current_value = r
        return r

    def bid(self):
        x = self.sample_value()
        if x > self.seuille:
            return x
        else:
            return (self.seuille - self.epsilon)*(1-self.distrib.cdf(self.seuille))/(1-self.distrib.cdf(x)) + self.epsilon

    def formula_bid(self, x):
        if x > self.seuille:
            return x
        else:
            return (self.seuille - self.epsilon)*(1-self.distrib.cdf(self.seuille))/(1-self.distrib.cdf(x)) + self.epsilon

    def virtual_value(self, x):
        if x < 0.5:
            return self.epsilon*x
        else:
            return x - (1 - self.distrib.cdf(x))/self.distrib.pdf(x)

    def update_revenue(self, payment):
        self.revenue += self.current_value - payment

    def define_bound(self):
            inf = 0
            sup = 1
            return [inf,sup]

    def define_bound_without_seuille(self):
        inf = self.distrib.ppf(0.01) * self.shading_factor[0] + self.shading_factor[1]
        sup = self.distrib.ppf(0.99) * self.shading_factor[0] + self.shading_factor[1]
        return [inf, sup]

class Seller:
    def __init__(self, type_auction, reserve_price_strategy, list_bidders, reserve_price = None):
        self.type_auction = type_auction
        self.reserve_price_strategy = reserve_price_strategy
        self.list_bidders = list_bidders
        self.reserve_price = self.define_reserve_price(reserve_price)
        self.revenue = 0
        self.nb_bidders = len(list_bidders)

    def define_reserve_price(self,reserve_price):
        if self.reserve_price_strategy == "no_reserve_price":
            reserve_price = [0.0 for _ in self.list_bidders]
        elif self.reserve_price_strategy == "hardcoded":
            reserve_price = reserve_price
        else:
            reserve_price = self.set_non_anonymous_reserve_price_analytically()
        return reserve_price

    def set_non_anonymous_reserve_price_analytically(self):
        reserve_price = [bidder.formula_bid(utils.compute_reserve_price(
        bidder.virtual_value, bidder.bound)) for bidder in self.list_bidders]
        return reserve_price

    def run_auction(self, list_bids):
        if self.type_auction == "second_price":
            winner = np.random.choice((np.flatnonzero(list_bids == max(list_bids))))
            if list_bids[winner] >= self.reserve_price[winner]:
                if self.nb_bidders > 1:
                    payment = max(np.sort(list_bids)[-2], self.reserve_price[winner])
                else:
                    payment = self.reserve_price[winner]
                self.list_bidders[winner].update_revenue(payment)
                self.revenue += payment
        elif self.type_auction == "first_price":
            if max(list_bids) == 0 :
                winner = int(np.random.choice(np.arange(0,len(list_bids),1.0)))
            else:
                winner = np.random.choice((np.flatnonzero(list_bids == max(list_bids))))
            if list_bids[winner] >= self.reserve_price[winner]:
                payment = list_bids[winner]
                self.list_bidders[winner].update_revenue(payment)
                self.revenue += payment
        else:
            print("wrong type auction")


def run_simu(seller, nb_simu):
    for i in range(nb_simu):
        list_bids = [bidder.bid() for bidder in seller.list_bidders]
        seller.run_auction(list_bids)

def experiment_ind_campgn_mgmt(list_shading, distrib, nb_runs, type_seller='second_price',reserve_price_strategy="monopoly",):
    list_bidders = [Bidder(distrib, shading) for shading in list_shading]
    seller = Seller(type_seller, reserve_price_strategy, list_bidders)
    run_simu(seller, nb_runs)
    exp_bidders_revenue = [bidder.revenue/nb_runs for bidder in list_bidders]
    exp_seller_revenue = seller.revenue/nb_runs
    return exp_bidders_revenue, exp_seller_revenue, seller.reserve_price

def experiment_ind_campgn_mgmt_seuille(list_shading, distrib, nb_runs, type_seller='second_price',reserve_price_strategy="monopoly", seuille = 0.5):
    list_bidders_seuille = [Bidder_seuille(distrib, seuille)]
    list_bidders = list_bidders_seuille + [Bidder(distrib, shading) for shading in list_shading[1:]]
    seller = Seller(type_seller,reserve_price_strategy, list_bidders)
    run_simu(seller, nb_runs)
    exp_bidders_revenue = [bidder.revenue/nb_runs for bidder in list_bidders]
    exp_seller_revenue = seller.revenue/nb_runs
    return exp_bidders_revenue, exp_seller_revenue, seller.reserve_price

def experiment_ind_campgn_mgmt_seuille_symmetric(list_shading, distrib, nb_runs, type_seller='second_price',reserve_price_strategy="monopoly", seuille = 0.5):
    list_bidders = [Bidder_seuille(distrib, seuille) for shading in list_shading]
    seller = Seller(type_seller, reserve_price_strategy, list_bidders)
    run_simu(seller, nb_runs)
    exp_bidders_revenue = [bidder.revenue/nb_runs for bidder in list_bidders]
    exp_seller_revenue = seller.revenue/nb_runs
    return exp_bidders_revenue, exp_seller_revenue, seller.reserve_price
