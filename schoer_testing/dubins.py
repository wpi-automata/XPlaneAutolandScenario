import numpy as np
from numpy import sin, cos, tan

from cbf_toolbox.dynamics import Dynamics
from cbf_toolbox.safety import Simulation
from cbf_toolbox.vertex import Agent

import matplotlib.pyplot as plt
from pprint import pprint

class FixedWingDubins3d(Dynamics):
    def __init__(self) -> None:
        n = 7
        m = 3

        def v_zeta(x):
            n,e,d,phi,theta,psi,V_T = x
            return np.array([V_T * cos(theta) * cos(psi),
                             V_T * cos(theta) * sin(psi),
                            -V_T * sin(theta)            ])

        def f_xi(x):
            g_D = 9.81
            n,e,d,phi,theta,psi,V_T = x
            return g_D/V_T * np.array([sin(phi)*cos(phi)*sin(theta),
                                      -(sin(phi)**2)*cos(theta),
                                       sin(phi)*cos(phi)            ])
        
        def g_xi(x):
            n,e,d,phi,theta,psi,V_T = x
            return np.array([[0, 1, sin(phi)*tan(theta)],
                             [0, 0, cos(phi)],
                             [0, 0, sin(phi)/cos(theta)]])

        f = lambda x: np.hstack([v_zeta(x), f_xi(x), 0])
        g = lambda x: np.vstack([np.zeros((3,3)), g_xi(x), np.array([1,0,0])])

        super().__init__(n, m, f, g)

        self.fig, self.axs = plt.subplots(3,2)

    def plot_point(self,x,t):
        n,e,d,phi,theta,psi,V_T = x
        # North by East
        self.axs[0,0].plot(e,n,'ob')
        self.axs[0,0].set_xlabel('East')
        self.axs[0,0].set_ylabel('North')
        # Up by East
        self.axs[1,0].plot(e,-d,'ob')
        self.axs[1,0].set_xlabel('East')
        self.axs[1,0].set_ylabel('Up')
        # Up by North
        self.axs[2,0].plot(n,-d,'ob')
        self.axs[2,0].set_xlabel('North')
        self.axs[2,0].set_ylabel('Up')

        # phi by time
        self.axs[0,1].plot(t,phi % 2*np.pi,'ob')
        self.axs[0,1].set_xlabel('time')
        self.axs[0,1].set_ylabel('$\phi$')

        # theta by time
        self.axs[1,1].plot(t,theta % 2*np.pi,'ob')
        self.axs[1,1].set_xlabel('time')
        self.axs[1,1].set_ylabel('$\\theta$')

        # psi by time
        self.axs[2,1].plot(t,psi % 2*np.pi,'ob')
        self.axs[2,1].set_xlabel('time')
        self.axs[2,1].set_ylabel('$\psi$')


if __name__ == '__main__':
    dyn = FixedWingDubins3d()
    x = np.array([0,0,-100,0,0,0,10])
    u = np.array([0,0,.5])
    dt = 0.1

    dyn.plot_point(x,0)
    
    for i in range(25):
        pprint(x)
        x = dyn.step(x,u,dt)
        dyn.plot_point(x,i+1)
    pprint(x)
    plt.show()