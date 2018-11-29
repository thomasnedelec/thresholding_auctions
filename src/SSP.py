from src.utils import *
import matplotlib.pyplot as plt

class SSP:
    def __init__(self, list_clients_represented, shading_factor=[1, 0]):
        self.list_clients_represented = list_clients_represented
        self.shading_factor = shading_factor
        self.nb_clients = len(list_clients_represented)
        self.current_client_represented = 0
        self.bound = self.define_bound()

    def bid(self):
        bids = [client.bid() for client in self.list_clients_represented]
        self.current_client_represented = np.argmax(bids)
        bid_SSP = self.shading_factor[0] * max(bids) + self.shading_factor[1]
        return bid_SSP

    def formula_bid(self, x):
        return self.shading_factor[0] * x + self.shading_factor[1]

    def update_revenue(self, payment):
        self.list_clients_represented[self.current_client_represented].update_revenue(payment)

    def plot_virtual_value(self):
        x = np.linspace(self.bound[0], self.bound[1], 100)
        plt.plot(x, self.virtual_value(x))
        plt.show()

    def virtual_value(self, x):
        cdf = lambda x: self.list_clients_represented[0].distrib.cdf(x) ** self.nb_clients
        pdf = lambda x: (self.nb_clients) * \
                        self.list_clients_represented[0].distrib.cdf(x) ** (self.nb_clients - 1) \
                        * self.list_clients_represented[0].distrib.pdf(x)

        return x - (1 - cdf((x - self.shading_factor[1]) / self.shading_factor[0])) / (
                pdf((x - self.shading_factor[1]) / self.shading_factor[0]) /
                self.shading_factor[0])

    def define_bound(self):
        inf = self.shading_factor[0] * min([bidder.bound[0] for bidder in self.list_clients_represented]) + \
              self.shading_factor[1]
        sup = self.shading_factor[0] * max([bidder.bound[1] for bidder in self.list_clients_represented]) + \
              self.shading_factor[1]
        print(inf)
        print(sup)
        return [inf, sup]


class SSP_seuille:
    def __init__(self, list_clients_represented, seuille=0.5):
        self.list_clients_represented = list_clients_represented
        self.nb_clients = len(list_clients_represented)
        self.current_client_represented = 0
        self.distrib = list_clients_represented[0].distrib
        self.bound = self.define_bound()
        self.epsilon = 0.001
        self.seuille = seuille

    def bid(self):
        bids = [client.bid() for client in self.list_clients_represented]
        self.current_client_represented = np.argmax(bids)
        max_bids = np.max(bids)
        if max_bids > self.seuille:
            bid_SSP = max_bids
        else:
            bid_SSP = (self.seuille - self.epsilon)*(1-self.distrib.cdf(self.seuille)**self.nb_clients)/\
                      (1-self.distrib.cdf(max_bids)**self.nb_clients) + self.epsilon
        return bid_SSP

    def formula_bid(self, x):
        if x > self.seuille:
            return x
        else:
            return (self.seuille - self.epsilon)*(1-self.distrib.cdf(self.seuille)**self.nb_clients)/\
                      (1-self.distrib.cdf(x)**self.nb_clients) + self.epsilon

    def update_revenue(self, payment):
        self.list_clients_represented[self.current_client_represented].update_revenue(payment)

    def plot_virtual_value(self):
        x = np.linspace(self.bound[0], self.bound[1], 100)
        plt.plot(x, self.virtual_value(x))
        plt.show()

    def virtual_value(self, x):
        return self.epsilon*x

    def define_bound(self):
        inf = 0.0
        sup = self.distrib.ppf(0.99999)
        return [inf, sup]