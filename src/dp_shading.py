import numpy as np
import itertools

def function_revenue(alpha, lambda_0, lambda_1,lambda_2):
    reserve_value = (alpha*lambda_0 + (1-alpha)*lambda_1)/(2*lambda_2)
    revenue = lambda_2*(1-2*lambda_2)*(1/3 - 1/3*reserve_value**3) + 1/2*(lambda_2**2)*(1 - reserve_value**2)
    return revenue


class DPShading:
    def __init__(self, horizon, discretisation, alpha, restriction_exploration= False):
        self.best_action_to_take = np.zeros((horizon, discretisation, discretisation))
        self.best_revenue_to_reach = np.zeros((horizon, discretisation, discretisation))
        self.horizon = horizon
        self.discretisation = discretisation
        self.shading_space = np.linspace(0.01, 1.0, discretisation)
        self.alpha = alpha
        self.restriction_exploration = restriction_exploration

    def run_backward_loop(self):
        for t in range(self.horizon):
            t = self.horizon - t - 1
            self.compute_revenue_given_t(t)

    def compute_revenue_given_t(self,t):
       if self.restriction_exploration:
           exploration1 = min(self.horizon - t + 1, self.discretisation)
           exploration2 = min(self.horizon - t + 2, self.discretisation)
           if t == self.horizon - 1:
               for i, j in itertools.product(range(exploration2), range(exploration1)):
                   self.compute_revenue_given_config_init(self.discretisation - i - 1, self.discretisation - j- 1)
           else:
                for i, j in itertools.product(range(exploration2), range(exploration1)):
                        self.update_matrix(t, self.discretisation - i - 1, self.discretisation - j - 1)

       else:
            if t == self.horizon -1:
                for i, j in itertools.product(range(self.discretisation ), range(self.discretisation)):
                    self.compute_revenue_given_config_init(i, j)
            else:
                for i,j in itertools.product(range(self.discretisation), range(self.discretisation)):
                    self.update_matrix(t, i, j)

    def compute_revenue_given_config_init(self, index_lambda_0, index_lambda_1):
        lambda_0 = self.shading_space[index_lambda_0]
        lambda_1 = self.shading_space[index_lambda_1]
        revenue = function_revenue(self.alpha, lambda_0, lambda_1, 1)
        self.best_revenue_to_reach[self.horizon-1,index_lambda_0, index_lambda_1] = revenue
        self.best_action_to_take[self.horizon-1,index_lambda_0, index_lambda_1] = self.discretisation - 1

    def update_matrix(self, t, index_lambda_0, index_lambda_1):
        if self.restriction_exploration:
            lambda_0 = self.shading_space[index_lambda_0]
            lambda_1 = self.shading_space[index_lambda_1]

            lambda_2 = self.shading_space[index_lambda_1]
            max_revenue = function_revenue(self.alpha, lambda_0, lambda_1, lambda_2) \
                                           + self.best_revenue_to_reach[t + 1, index_lambda_1, index_lambda_1]
            index_max_revenue = index_lambda_1

            if index_lambda_1 > 0:
                lambda_2 = self.shading_space[index_lambda_1 - 1]
                revenue = function_revenue(self.alpha, lambda_0, lambda_1, lambda_2) \
                                  + self.best_revenue_to_reach[t+1,index_lambda_1, index_lambda_1 - 1]
                if revenue > max_revenue:
                    max_revenue = revenue
                    index_max_revenue = index_lambda_1 - 1

            if index_lambda_1 < self.discretisation - 2:
                lambda_2 = self.shading_space[index_lambda_1 + 1]
                revenue = function_revenue(self.alpha, lambda_0, lambda_1, lambda_2) \
                                               + self.best_revenue_to_reach[t + 1, index_lambda_1, index_lambda_1 + 1]
                if revenue > max_revenue:
                    max_revenue = revenue
                    index_max_revenue = index_lambda_1 + 1
            self.best_revenue_to_reach[t, index_lambda_0, index_lambda_1] = max_revenue
            self.best_action_to_take[t, index_lambda_0, index_lambda_1] = index_max_revenue

        else:
            revenue_list = np.zeros(self.discretisation)
            for i in range(self.discretisation):

                lambda_0 = self.shading_space[index_lambda_0]
                lambda_1 = self.shading_space[index_lambda_1]
                lambda_2 = self.shading_space[i]

                revenue_list[i] = function_revenue(self.alpha, lambda_0, lambda_1, lambda_2) \
                                  + self.best_revenue_to_reach[t+1,index_lambda_1, i]

            self.best_revenue_to_reach[t, index_lambda_0, index_lambda_1] = np.max(revenue_list)
            self.best_action_to_take[t, index_lambda_0, index_lambda_1] = np.argmax(revenue_list)

    def extract_best_path_from_truthful(self):
        path = []
        current_best_state_0 = int((self.discretisation - 1)/2)
        path.append(self.shading_space[current_best_state_0])

        decision= np.argmax(self.best_revenue_to_reach[0, current_best_state_0, current_best_state_0-1:current_best_state_0+1])
        current_best_state_1 = current_best_state_0 - 1 + decision
        print(current_best_state_1)
        for t in range(self.horizon):
            path.append(self.shading_space[int(current_best_state_1)])
            next_step = int(self.best_action_to_take[t,current_best_state_0, current_best_state_1])
            current_best_state_0 = current_best_state_1
            current_best_state_1 = next_step

        return path
