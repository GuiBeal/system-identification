import numpy as np
import control
from pysid import arx

z = control.TransferFunction.z
G = (z - 0.8) / (z - 0.9) / (z - 0.4)

k = np.arange(0, 20+1, 1)
u = np.sin(0.5*k) + np.sin(0.25*k)

y = control.forced_response(G, U=u, return_x=False)[1]

G_arx = arx(2, 2, 1, u, y)

A = G_arx.A[0][0]
B = G_arx.B[0][0]

display(A)
display(B)
