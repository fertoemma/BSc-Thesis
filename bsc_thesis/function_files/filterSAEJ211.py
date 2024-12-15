#filterSAEJ211.py
"""
Generic SAE J211 filter

Created by Gergely ERDOS and Csaba MATZON (eCon Engineering)

Contact: gergely.erdos@econengineering.com, csaba.matzon@econengineering.com
"""

import numpy as np
import math

def filterSAEJ211(data, CFC, dt):
    pad = int(round(10E-3/dt))
    startpad = list(range(pad))
    endpad = list(range(pad))
    mstart = data[0]
    mend = data[-1]
    for i in range(pad):
        startpad[pad-i-1] = mstart - (data[i+1] - mstart)
        endpad[i] = mend - (data[-i-2] - mend)
    data = np.array(list(startpad) + list(data) + list(endpad))

    omd = float(2*math.pi*CFC*2.0775)
    oma = float(math.sin(omd*dt/2) / math.cos(omd*dt/2))
    a0 = oma**2 / (1.0 + math.sqrt(2)*oma + oma**2)
    a1 = 2*a0
    a2 = a0
    b1 = -2*(oma**2 - 1) / (1.0 + math.sqrt(2)*oma + oma**2)
    b2 = (-1 + math.sqrt(2)*oma - oma**2) / (1.0 + math.sqrt(2)*oma + oma**2)

    #pass1 forward
    ndata = [a0*data[0], a0*data[1] + a1*data[0] + b1*(a0*data[0])]
    for i in range(len(data)-2):
        a = i+2
        ndata.append(a0*data[a] + a1*data[a-1] + a2*data[a-2] + b1*ndata[-1] + b2*ndata[-2])
    ndata.reverse()
    data = ndata

    #pass2 reverse
    ndata = [a0*data[0], a0*data[1] + a1*data[0] + b1*(a0*data[0])]
    for i in range(len(data)-2):
        a = i+2
        ndata.append(a0*data[a] + a1*data[a-1] + a2*data[a-2] + b1*ndata[-1] + b2*ndata[-2])
    ndata.reverse()
    data = ndata

    data = data[pad:-pad]
    return data
