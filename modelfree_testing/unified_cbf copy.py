import os
import sys
CBF_DIR = os.environ['CBF_TOOLBOX_DIR']
sys.path.append(CBF_DIR)

from cbf_toolbox.dynamics import Dynamics, SingleIntegrator
from cbf_toolbox.safety import Simulation
from cbf_toolbox.vertex import Agent, Obstacle
from cbf_toolbox.edge import CBF

import math
import numpy as np
from numpy import sin, cos, tan
from jax import grad
import jax.numpy as jnp
from dubins import FixedWingDubins3d
from cbf_toolbox.geometry import HalfPlaneND


class UnifiedCBF():
    def __init__(self, kappa, agent, dynamics, obstacle_list):
        self.kappa = kappa
        self.agent = agent
        self.obstacle_list = obstacle_list
        self.dynamics = dynamics

        def get_h(obstacle):
            x_agent = self.agent.state
            u_agent = self.agent.u
            x_obs = obstacle.state
            u_obs = obstacle.u
            x = x_agent - x_obs
            xdot = self.agent.dynamics.dx(x_agent,u_agent) - obstacle.dynamics.dx(x_obs,u_obs)
            agent_rad = self.agent.shape.radius

            barrier = obstacle.shape.func

            h = barrier(x,agent_rad)

            
            # grad_h = np.array(grad(barrier, argnums=0)(x,agent_rad))

            # hdot = grad_h * xdot = grad_h * (f + g*u) = Lf_h + Lg_h*u
            # hdot = grad_h.T.dot(xdot)
            return h, x
        

        def unify(obstacle_list):
            x_agent = self.agent.state
            u_agent = self.agent.u
            h = 0
            for obstacle in obstacle_list:
                barrier = obstacle.shape.func
                x = x_agent - obstacle.state
                h_i = barrier(x,offset)

                # h_i, x= get_h(obstacle)
                h += jnp.exp(-self.kappa * h_i)
                
            return -1/self.kappa * jnp.log(h)                    
       
        
        
        x_agent = self.agent.state
        u_agent = self.agent.u
        xdot = self.agent.dynamics.dx(x_agent,u_agent)
        offset = self.agent.shape.radius

        uni = lambda x,rad : jnp.sum([o.shape.func(x,rad) for o in obstacle_list])
        grad_uni = np.array(grad(uni)(x_agent,offset))

        self.barrier_unif = unify(obstacle_list)
        grad_h = np.array(grad(unify)(obstacle_list))
        print(grad_h.T.dot(xdot))
        

def safe_vel(r, v_d, gamma_v=4, sigma=3, gamma_p=0.1):
    P_v = v_d.dot(v_d.T)/np.linalg.norm(v_d)**2
    W_v = P_v + 1/np.sqrt(gamma_v)*(np.eye(3) - P_v)
    Gamma_v = P_v + gamma_v*(np.eye(3) - P_v)

    b_v = grad_h * W_v

    alpha_p = gamma_p*r
    a_v = hdot + alpha_p * h - sigma*grad_h.dot(grad_h.T)
    v_s = v_d + Lamda(a_v, np.linalg.norm(b_v)) * W_v*b_v.T

    return v_s
    
def Lamda(a, b, nu_v = 0.007):
    # eq. 21
    if b == 0:
        return 0
    
    return 1/nu_v**b * np.ln(1+math.exp(-nu_e * a/b))
        
def clf(v_zeta, v_c, R, R_d, mu=1e-4):
    #
    V = 0.5*(v_c - v_zeta).T.dot(v_c - v_zeta) + 1/(2*mu)*(R-R_d)**2

class TrackingController:
    def __init__(self, x, t, 
                 Kv=0.3*np.eye(3), v_g=np.array([0,161.32,0]).T, Kr = 0.05*np.eye(3), mu=1e-5, lamda=0.2, g_D=9.81): 
        # v_c = v_d  (velocity command = desired velocity)
        # goal: track v_d, output u = [A_T, P, Q] that is exponentially stable


        # a_d = a_c + 1/2 * Kv*(v_c - rdot) 
        n,e,d,phi,theta,psi,V_T = x
        R = g_D/V_T * sin(psi)*cos(theta)
        r = np.array([n, e, d]).T
        v_zeta = np.array([V_T * np.cos(theta) * cos(psi),
                             V_T * cos(theta) * sin(psi),
                            -V_T * sin(theta)            ])
        
        r_g = v_g*t
        v_d = v_g + Kr*(r_g - r)

        v_c = v_d
        a_d = a_c + 1/2 * Kv*(v_c - v_zeta)

        # TODO: a_c??

        # TODO: figure out P CLF. also, this only tracking model-free. what about actruall model-free?

        sphi, cphi = [sin(phi), cos(phi)]
        stheta, ctheta = [sin(theta), cos(theta)]
        spsi, cpsi = [sin(psi), cos(psi)]
        M_a = np.array([[ctheta*cpsi, -V_T*(cphi*stheta*cpsi + sphi*spsi),  V_T*(sphi*stheta*cpsi - cphi*spsi)],
                        [ctheta*spsi,  V_T*(-cpsi*stheta*spsi + sphi*cpsi), V_T*(sphi*stheta*spsi + cphi*cpsi)],
                        [-stheta,     -V_T*cphi*ctheta,                     V_T*sphi*ctheta]
                       ])
        A_T, Q, R_d = (np.linalg.inv(M_a)*a_d).T
        M_R = M_a[:, -1]
        
        # actual acceleration
        vdot = a_d + M_R*(R-R_d)

        # TODO: P
    
    



if __name__ == '__main__':
    dyn = FixedWingDubins3d()
    n = 7
    m = 3
    # stationary dynamics
    f = lambda x : np.zeros(n) # Drift function
    g = lambda u : np.zeros([n,m]) # Control function
    static_dyn = Dynamics(n, m, f, g)

    # model-free dynamics: single integrator
    dyn_modelfree = SingleIntegrator(n)
    s = Simulation()

    x = np.array([0,0,-100,0,0,0,10])
    u = np.array([0,0,.5])
    bound = np.array([[0,0],[-10.,10.]]) # dummy parameter, doesn't actually mean anything
    
    agent = Agent(x, 0.5, dyn_modelfree,plot_path=False,plot_arrows=False)
    s.add_agent(agent,u)

    spec1_geq = HalfPlaneND(n = np.array([1.,0.,0.,0.,0.,0.,0.]), d = -2, geq = 1, bound=bound)
    spec1_leq = HalfPlaneND(n = np.array([1.,0.,0.,0.,0.,0.,0.]), d = 2, geq = -1, bound=bound)
    
    o1_geq = Obstacle(state=spec1_geq.state, shape=spec1_geq, dynamics=static_dyn)
    o1_leq = Obstacle(state=spec1_leq.state, shape=spec1_leq, dynamics=static_dyn)

    kappa = 0.5
    cbf = UnifiedCBF(kappa, agent, dyn_modelfree, [o1_geq, o1_leq])
    print(cbf.barrier_unif)
