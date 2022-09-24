"Kaiser, J., Mostafa, H., & Neftci, E. (2020). Synaptic plasticity dynamics for deep continuous local learning (DECOLLE). Frontiers in Neuroscience, 14, 424."

class DECOLLE_Neuron(nn.Module):
    def __init__(
        self, 
        layer_block,
        spike_grad=snntorch.surrogate.fast_sigmoid(slope=10),
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

class snnmodel(nn.Module):
    def __init__(self, kernel_size, spike_grad=snntorch.surrogate.fast_sigmoid(slope=10)):
        super(snnmodel, self).__init__()
        
        self.deco1 = DECOLLE_Neuron(nn.Conv2d(2, 64, kernel_size, stride=1, padding=3),alpha=0.97, beta=0.92,alpharp=0.65)
        self.deco2 = DECOLLE_Neuron(nn.Conv2d(64, 128, kernel_size, stride=1, padding=3),alpha=0.97, beta=0.92,alpharp=0.65)
        self.deco3 = DECOLLE_Neuron(nn.Conv2d(128, 128, kernel_size, stride=1, padding=3),alpha=0.97, beta=0.92,alpharp=0.65)

        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(1)
        self.maxpool3 = nn.MaxPool2d(2)
        
        self.lif1 = spike_grad
        self.lif2 = spike_grad
        self.lif3 = spike_grad
        
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        
        self.fc1 = nn.Linear(16*16*64, 11)
        self.fc2 = nn.Linear(16*16*128, 11)
        self.fc3 = nn.Linear(8*8*128, 11)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        
    def forward(self, x, reset):
        
        spk_out = []
        r_out = []
        mem_out = []
        
        # stage 1
        
        cur = self.deco1(x, reset)
        pool = self.maxpool1(cur)
        spk = self.lif1(pool)
        dspk = self.dropout1(spk)
        flat = self.flatten1(dspk)
        r = self.fc1(flat)
               
        spk_out.append(spk)
        r_out.append(r)
        mem_out.append(pool)
        
        # stage 2
        
        x = spk.detach_()
        cur = self.deco2(x, reset)
        pool = self.maxpool2(cur)
        spk = self.lif2(pool)
        dspk = self.dropout2(spk)
        flat = self.flatten2(dspk)
        r = self.fc2(flat)
        
        spk_out.append(spk)
        r_out.append(r)
        mem_out.append(pool)
        
        # stage 3
        
        x = spk.detach_()
        cur = self.deco3(x, reset)
        pool = self.maxpool3(cur)
        spk = self.lif3(pool)
        dspk = self.dropout3(spk)
        flat = self.flatten3(dspk)
        r = self.fc3(flat)
        
        spk_out.append(spk)
        r_out.append(r)
        mem_out.append(pool)

        return spk_out, r_out, mem_out
