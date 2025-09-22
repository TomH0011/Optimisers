import sys



class Optimisers:
    def __init__(self):
        self.velocities = None
        self.TorchTensor = getattr(sys.modules.get("torch"), "Tensor", None)

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

    def SGD(self, parameters, gradients, learning_rate, steps=None, epsilon=None, weight_decay=None):

        # Helper function
        def rebuild(old_data, new_data):
            return type(old_data)(new_data)

        if not isinstance(parameters, (list, tuple, self.TorchTensor)):
            raise TypeError("Parameters must be list, tuple, or torch.Tensor")

        new_params = []
        for p, g in zip(parameters, gradients):
            if weight_decay:
                g = g + weight_decay * p
            new_p = p - learning_rate * g
            new_params.append(new_p)

        return rebuild(parameters, new_params)

    def SGDMA(self, parameters, gradients, momentum, learning_rate, steps=None, epsilon=None, weight_decay=None):
        """
        Stochastic Gradient Descent with Momentum and Acceleration (Nesterov).
        parameters : list | tuple | torch.Tensor
        gradients  : same structure as parameters, gradient at the *lookahead point*
        """

        def rebuild(old_data, new_data):
            return type(old_data)(new_data)

        if not isinstance(parameters, (list, tuple, self.TorchTensor)):
            raise TypeError("Parameters must be list, tuple, or torch.Tensor")

        new_params = []
        new_velocities = []

        # Initialise velocities if missing
        if not hasattr(self, "velocities") or len(self.velocities) != len(parameters):
            self.velocities = [0 for _ in parameters]

        for p, g, v in zip(parameters, gradients, self.velocities):
            if weight_decay:
                g = g + weight_decay * p

            # Nesterov momentum update
            p_look = p + momentum * v  # lookahead
            # g is expected to be computed at p_look externally

            v = momentum * v - learning_rate * g
            new_p = p + v

            new_params.append(new_p)
            new_velocities.append(v)

        self.velocities = new_velocities  # store velocities

        return rebuild(parameters, new_params)

    def NAG(self, parameters, gradients, momentum, acceleration, learning_rate, steps=None, epsilon=None, weight_decay=None):

        # Helper function
        def rebuild(old_data, new_data):
            return type(old_data)(new_data)

        if not isinstance(parameters, (list, tuple, self.TorchTensor)):
            raise TypeError("Parameters must be list, tuple, or torch.Tensor")

        new_params = []
        velocity = 0
        lookahead_velocity = 0
        for p, g in zip(parameters, gradients):
            if weight_decay:
                g = g + weight_decay * p
            if velocity > 0:
                velocity = momentum * velocity - learning_rate * g
                # Remember p is shape [batch_size, embedding_dim]
                new_p = p + velocity
            else:
                new_p = p - learning_rate * g
            new_params.append(new_p)

        return rebuild(parameters, new_params)


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

    def LRTAstar(self):
        return
