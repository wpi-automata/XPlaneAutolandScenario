import os
import sys
CBF_DIR = os.environ['CBF_TOOLBOX_DIR']
sys.path.append(CBF_DIR)
sys.path.append(os.getcwd())


from cbf_toolbox.dynamics import Dynamics, SingleIntegrator
from cbf_toolbox.safety import Simulation
from cbf_toolbox.vertex import Agent, Obstacle
from cbf_toolbox.edge import CBF, ReferenceControl
from cbf_toolbox.geometry import HalfPlaneND
from src.xplane_autoland.utils.pid import PID
from dubins import FixedWingDubins3d

import math
import numpy as np
from numpy import sin, cos, tan
from jax import grad
import jax.numpy as jnp
        

class TrackingController:
    def __init__(self, des_v=5., dt=0.1):
        self._des_v       = des_v # desired forward velocity
        self._dt          = dt

        if dt > 0.5:
            raise Warning("Running at a much slower dt than controller was designed for")

        # PI controllers
        self._v_pid     = PID(dt, kp=50., ki=5., kd=0.)
        self._ndot_pid   = PID(dt, kp=0.24, ki=0.024, kd=0.)
        self._edot_pid = PID(dt, kp=0.24, ki=0.024, kd=0.)
        self._ddot_pid   = PID(dt, kp=0.24, ki=0.024, kd=0.)
    
    def control(self, v_s):
        '''
        INPUTS
            safe velocity [ndot, edot, ddot].T
            
        OUTPUTS
            controls 
            A_T     acceleration 
            P       ang velocity for roll
            Q       ang velocity for pitch
        '''

        ndot, edot, ddot = v_s.flatten()

        err_ndot = 0.0 - ndot
        err_edot = 0.0 - edot
        err_ddot = 0.0 - ddot
        err_a = self._des_v - V_T

        A_T = self._v_pid(err_a)
        P = self._ddot_pid(err_ddot) + self._ddot_pid(err_edot)
        Q = -self._ddot_pid(err_ddot) + self._ndot_pid(err_ndot)
        
        return A_T, P, Q
    
class SingleIntController:
    def __init__(self, des_v=50., dt=0.1):
        self._des_v       = des_v # desired forward velocity
        self._dt          = dt

        if dt > 0.5:
            raise Warning("Running at a much slower dt than controller was designed for")

        # PI controllers
        self._n_pid   = PID(dt, kp=0.24, ki=0.024, kd=0.)
        self._e_pid = PID(dt, kp=0.24, ki=0.024, kd=0.)
        self._d_pid   = PID(dt, kp=0.24, ki=0.024, kd=0.)
    
    def control(self, statevec):
        '''
        INPUTS
            n, e, d = position

        OUTPUTS
            controls: ndot, edot, ddot
            
        '''

        n, e, d = statevec

        err_n = 0.0 - n
        err_e = 0.0 - e
        err_d = 0.0 - d

        ndot = self._n_pid(n)
        edot = self._e_pid(e)
        ddot = self._d_pid(d)
        
        return np.array([ndot, edot, ddot]).T


def get_safe_vel(r, v_d, grad_h, hdot, gamma_v=4, sigma=3, gamma_p=0.1):
    r = r[np.newaxis].T
    v_d = v_d[np.newaxis].T
    grad_h = grad_h[np.newaxis].T

    P_v = v_d.dot(v_d.T)/np.linalg.norm(v_d)**2
    W_v = P_v + 1/np.sqrt(gamma_v)*(np.eye(3) - P_v)
    Gamma_v = P_v + gamma_v*(np.eye(3) - P_v)

    b_v = grad_h.T.dot(W_v)

    alpha_p = gamma_p*r
    a_v = hdot + alpha_p * h - sigma*grad_h.dot(grad_h.T)
    v_s = v_d + (Lambda(a_v, np.linalg.norm(b_v)) * W_v).dot(b_v.T)

    return v_s
    
def Lambda(a, b, nu_v = 0.007):
    # eq. 21
    if b == 0:
        return 0
    
    return 1/nu_v**b * np.log(1+np.exp(-nu_v * a/b))
      
def update_sim(sim, u_ref, state):
    # update reference controls in sim 
    for a in s.agents:
        a.u_ref = u_ref
        a.goal = None
        a.state = state
    
    for idx in range(len(sim.control)):
        control = sim.control[idx]
        if isinstance(control,ReferenceControl):
            sim.control[idx] = ReferenceControl(sim.agents[0],np.array(sim.agents[0].u_ref))

def get_grad_h(sim, state, u, agent_rad):
    agent = sim.agents[0]
    obstacle = sim.obsts[0]
    x_agent = state
    u_agent = u
    x_obs = obstacle.state
    u_obs = obstacle.u
    x = x_agent - x_obs
    xdot = agent.dynamics.dx(x_agent,u_agent) - obstacle.dynamics.dx(x_obs,u_obs)
    agent_rad = agent.shape.radius

    h_func = obstacle.shape.func

    h = h_func(x,agent_rad)
    grad_h = np.array(grad(h_func, argnums=0)(r,agent_rad))
    hdot = grad_h.T.dot(xdot)

    return h, grad_h, hdot

        

if __name__ == '__main__':
    
    # initializing
    dyn = FixedWingDubins3d()
    x = np.array([0.,0.,-100.,0.,0.,0.,10.])
    u = np.array([0.,0.,.5])
    dt = 0.1
    agent_rad = 1
    s = Simulation()
    singleInt_c = SingleIntController(dt=dt)
    track_c = TrackingController(dt=dt)

    # ==== model-free dynamics: single integrator with position r ==== 
    dyn_modelfree = SingleIntegrator(3)
    r = np.array(x[:3])
    agent = Agent(r, 0.5, dyn_modelfree,plot_path=False,plot_arrows=False)
    s.add_agent(agent,u)

    # ==== CBF: single half-plane ==== 
    # stationary dynamics
    f = lambda x : np.zeros(3) # Drift function
    g = lambda u : np.zeros(3) # Control function
    static_dyn = Dynamics(3, 3, f, g)

    cbf = HalfPlaneND(n = np.array([1.,0.,0.]), d = -2, geq = 1, bound=np.array([[0,0],[0,0]]))
    obs = Obstacle(state=cbf.state, shape=cbf, dynamics=static_dyn)
    
    h = obs.shape.func
    s.add_obstacle(obst=obs)
    
    max_time = 100
    for step in range(math.ceil(max_time/dt)):
        n,e,d,phi,theta,psi,V_T = x
        r = np.array([n, e, d])

        # # velocity tracking controller to get single integrator controls
        v_d = singleInt_c.control(r)

        s.ready_to_sim = False
        update_sim(s, v_d, r)
        s.simulate(num_steps=1, dt = dt, plotting=False)

        # ====== convert desired velocity into safe velocity =====
        # model-free desired velocity 
        
        h, grad_h, hdot = get_grad_h(s, r, v_d, agent_rad)
        
        v_s = get_safe_vel(r, v_d, grad_h, hdot)
        
        print('safe', v_s)

        # tracking controller convert v_s into controls  u_s = [A_T P Q]
        u_s = track_c.control(v_s)

        # get next state
        x = dyn.step(x, u_s)
        print('x', x)
        
