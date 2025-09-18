from config import learning_rate


class Optimisers:
    def __init__(self):
        pass

    def zero_grad(self, gradients):
        """
        gradients: same structure as parameters
        returns: zeroed gradients
        """
        return [0.0 * g for g in gradients]

    def GD(self, parameters, gradients, learning_rate, steps=None, epsilon=None, weight_decay=None):
        new_params = []
        for p, g in zip(parameters, gradients):
            if weight_decay:
                g = g + weight_decay * p
            new_p = p - learning_rate * g  # move away from higher Cost
            new_params.append(new_p)
        return new_params

    def SGD(self):
        return

    def mbGD(self):
        return

    def SGDM(self):
        return

    def SGDMA(self):
        return

    def AdaGrad(self):
        return

    def AdaDelta(self):
        return

    def AdaMax(self):
        return

    def Adam(self):
        return

    def NAdam(self):
        return

    def AdamW(self):
        return

    def NAG(self):
        return

    def AMSG(self):
        return

    def RMSProp(self):
        return

    def RAdam(self):
        return

    def Sophia(self):
        return

    def LARS(self):
        return

    def LAMB(self):
        return

    def AdaBelief(self):
        return

    def Lion(self):
        return

    def Shampoo(self):
        return
