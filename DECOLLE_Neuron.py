"Kaiser, J., Mostafa, H., & Neftci, E. (2020). Synaptic plasticity dynamics for deep continuous local learning (DECOLLE). Frontiers in Neuroscience, 14, 424."

class DECOLLE_Neuron(nn.Module):
    def __init__(
        self, 
        layer_block,
        spike_grad=surrogate.fast_sigmoid(slope=10),
        alpha=0.9, 
        beta=0.85, 
        alpharp=0.65, 
        wrp=1.0, 
        do_detach=True, 
        gain=1
        ):
        super(DECOLLE_Neuron, self).__init__()
        self.conv_layer = layer_block
        self.spike_grad = spike_grad

        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.beta = torch.tensor(beta, requires_grad=False)
        self.alpharp = torch.tensor(alpharp, requires_grad=False)
        self.wrp = torch.tensor(wrp, requires_grad=False)
        self.gain = torch.tensor(gain, requires_grad=False)
        
        self.do_detach = do_detach
        self.state = None
        self.P = torch.zeros(1)
        self.Q = torch.zeros(1)
        self.R = torch.zeros(1)
        self.S = torch.zeros(1)
        
    def forward(self, x, init_state=False):
        if init_state:
            self.state = None
        if self.state == None:
            P = torch.zeros_like(x)
            Q = torch.zeros_like(x)
            Q = self.beta * Q + (1-self.beta) * self.gain * x
            P = self.alpha * P + (1-self.alpha) * Q
            U = self.conv_layer(P)
            R = torch.zeros_like(U)
            S = torch.zeros_like(U)
            R = self.alpharp * R - (1-self.alpharp) * S * self.wrp
            U = U + R
            S = self.spike_grad(U)
            self.state = True
        else:
            P = self.P
            Q = self.Q
            R = self.R
            S = self.S
            Q = self.beta * Q + (1-self.beta) * self.gain * x
            P = self.alpha * P + (1-self.alpha) * Q
            R = self.alpharp * R - (1-self.alpharp) * S * self.wrp
            U = self.conv_layer(P) + R
            S = self.spike_grad(U)
            
        self.P = P
        self.Q = Q
        self.R = R
        self.S = S
        
        if self.do_detach:
            self.P.detach_()
            self.Q.detach_()
            self.R.detach_()
            self.S.detach_()
            
        return U
