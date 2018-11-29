class Bidder_first_price:
    def __init__(self, distrib, g, G, shading_factor= [1, 0],reserve_price=None):
        self.distrib = distrib
        self.shading_factor = shading_factor
        self.current_value = 0
        self.bound = self.define_bound()
        self.revenue = 0
        self.reserve_price = reserve_price
        self.g = g
        self.G = G

    def sample_value(self):
        r = self.distrib.rvs()
        self.current_value = r
        return r

    def bid(self):
        x = self.sample_value()
        if x < self.reserve_price:
            bid = 0
        if x >= self.reserve_price:
            bid = (self.reserve_price*self.G(self.reserve_price) +\
                  integrate.quad(lambda t: t*self.g(t), self.reserve_price, x)[0]) / self.G(x)
        #print(f"x:{x}")
        #print(f"bid:{bid}")
        return bid

    def formula_bid(self, x):
        return self.shading_factor[0] * x + self.shading_factor[1]

    def virtual_value(self, x):
        return self.formula_bid(x) - self.shading_factor[0] * (1 - self.distrib.cdf(x)) / (
                               self.distrib.pdf(x))

    def plot_virtual_value(self):
        x = np.linspace(self.bound[0],self.bound[1],100)
        plt.plot(x, self.virtual_value(x))
        plt.show()

    def update_revenue(self, payment):
        self.revenue += self.current_value - payment

    def formula_bid(self, x):
            return self.shading_factor[0] * x + self.shading_factor[1]

    def define_bound(self):
            inf = self.distrib.ppf(0.01)
            sup = self.distrib.ppf(0.99)
            return [inf,sup]
