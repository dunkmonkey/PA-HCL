import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16

x = torch.randn(batch,length,dim).to("cuda")

model = Mamba(

	# this module uses roughly 3 * expand *d model^2 parameters
	d_model=dim, # Model dimension d model
	d_state=16,  # SSM state expansion factor
	d_conv=4,    # Local convolution width
	expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
print(y.shape)
assert y.shape ==x.shape