import numpy as np
from numpy import sin, cos, tan
import matplotlib.pyplot as plt

import os
import sys
CBF_DIR = os.environ['CBF_TOOLBOX_DIR']
sys.path.append(CBF_DIR)
from cbf_toolbox.dynamics import Dynamics
from cbf_toolbox.vertex import Shape

from cbf_toolbox.safety import Simulation3d
from cbf_toolbox.vertex import Agent, Goal, Obstacle
from cbf_toolbox.geometry import HalfPlaneND

from cbf_toolbox.edge import ReferenceControl, CLF, CBF

class Airplane_upd(Dynamics):
    '''
    using state-space calculated in matlab
    Combining longitudinal and lateral into one state-space model

    state has components 
        u       velocity in x-dir
        v       velocity in y-dir
        w       velocity in z-dir
        p       roll rate
        q       pitch rate
        r       yaw rate
        phi     roll angle
        theta   pitch angle
        psi     yaw angle
        x       horizontal displacement
        y       lateral deviation wrt runway
        h       altitude

        control input has components
        delta_e      elevator deflection
        delta_t      throttle control (between 0 and 1)
        delta_a      aileron stabilizer deflection 
        delta_r       rudder deflection
    '''
    def __init__(self, dt=0.01):
        self._dt = dt
        self._g = 9.8  # Down gravity  (m/s^2)

        # [u; v; w; p; q; r; phi; theta; psi; x; y; h]
        n = 12 

        # [delta_e; delta_t; delta_a; delta_r]
        m = 4

        f = lambda x: np.array([
                                [-1.510152e-01,-8.438806e-05,-6.167490e-03,-5.835096e-05,-1.077133e-03,-2.911320e-04,-5.146673e-05,-9.772380e+00,-1.355497e-04,-2.194634e-04,-1.308875e-05,6.805497e-04],
                                [-1.000370e-05,-1.228001e-01,-1.325890e-07,-2.265612e+00,-6.430984e-06,3.672176e-03,-5.138398e+00,2.835131e-06,1.738803e+01,-1.415261e-05,1.701989e-02,1.038397e-04],
                                [3.022820e-02,1.477230e-05,-5.951605e-02,1.049373e-04,-4.785416e-01,2.732808e-04,7.172729e-05,9.403716e-01,1.844470e-04,-1.504789e-04,-2.021810e-05,1.966004e-04],
                                [-2.973816e-05,-3.875312e-02,3.684591e-05,-6.976174e-01,-3.984033e-06,-4.881639e-04,-9.223652e-01,2.757486e-05,2.284882e+00,1.507029e-04,-1.361071e-03,9.845046e-05],
                                [-6.734265e-04,-9.231071e-07,3.479465e-02,2.798387e-06,-8.502739e-02,-1.346463e-04,-9.472643e-06,-4.010334e-01,-2.151444e-05,1.425806e-04,1.260178e-05,-9.733455e-05],
                                [1.109235e-05,-3.222983e-03,-6.821018e-06,-6.597679e-02,-1.793900e-07,-2.437395e-04,-3.103770e-01,-5.497679e-07,9.222798e-01,-1.224643e-05,1.165120e-03,3.459610e-05],
                                [-4.738982e-05,3.891019e-03,2.446387e-05,5.619679e-02,-2.891013e-05,8.641949e-05,-6.868907e-01,2.354279e-04,-4.475528e-01,7.219427e-06,-2.018621e-02,-5.647148e-05],
                                [-1.102975e-04,-8.465882e-07,-3.361036e-05,6.335154e-06,3.219527e-05,3.584443e-05,2.632961e-06,1.000754e+00,1.420265e-05,-6.135789e-06,2.906449e-05,-3.983448e-05],
                                [-1.178841e-05,9.186544e-04,-8.921513e-06,1.437137e-02,3.406725e-05,1.076521e-04,-1.763956e-01,-7.579374e-05,-1.146790e-01,5.703121e-05,-5.178546e-03,-1.220799e-05],
                                [9.949321e-01,1.490221e-06,1.538135e-02,9.567196e-06,1.316371e-01,2.505559e-06,-4.659453e-07,-1.075419e-02,4.593896e-06,7.543382e-05,6.662100e-05,5.991756e-04],
                                [-9.589260e-05,1.519097e-01,1.294410e-05,3.098261e+00,-7.738550e-06,1.031541e-03,2.218044e+01,-3.481805e-06,-1.052013e+01,4.014826e-06,6.091794e-02,5.577779e-04],
                                [8.192225e-02,1.478060e-05,1.440399e-02,-1.396714e-05,-5.916046e-02,-6.601807e-05,-6.280880e-06,4.986285e+01,-4.466567e-05,-3.823779e-05,-3.972863e-06,8.845144e-04],
                                ]) @ x

        
        g = lambda x: np.array([
                                [-6.798975e-05,5.603672e-02,-1.338373e-04,-4.554871e-05],
                                [2.274398e-07,2.864630e-04,-1.731800e+03,5.045000e+03],
                                [-1.329134e-06,-5.760576e-06,2.166550e-04,7.385568e-05],
                                [-1.001282e-07,-1.737429e-05,9.669989e+01,-2.856000e+02],
                                [-3.450001e+01,4.050765e-05,5.951264e-06,2.106717e-06],
                                [-7.503982e-08,-9.566660e-07,9.999952e+00,-3.050002e+01],
                                [8.270046e-07,-2.526827e-06,-1.699962e+00,5.000014e+00],
                                [-1.144645e-04,-3.801312e-04,-4.133321e-07,1.326242e-07],
                                [1.203168e-06,-7.381879e-06,1.429760e-05,4.797558e-06],
                                [0,0,0,0],
                                [0,0,0,0],
                                [0,0,0,0],

                                ])
        
        super().__init__(n, m, f, g, dt)

    @property
    def state(self):
        return (self._rn, self._re, self._rd, self._psi)
    
    def __repr__(self):
        return f'Airplane'
    
class Airplane(Dynamics):
    '''
    Combining longitudinal and lateral into one state-space model

    state has components 
        u       velocity in x-dir
        v       velocity in y-dir
        w       velocity in z-dir
        p       roll rate
        q       pitch rate
        r       yaw rate
        phi     roll angle
        theta   pitch angle
        psi     yaw angle
        x       horizontal displacement
        y       lateral deviation wrt runway
        h       altitude

        control input has components
        delta_e      elevator deflection
        delta_t      throttle control (between 0 and 1)
        delta_a      aileron stabilizer deflection 
        delta_r       rudder deflection
    '''
    def __init__(self, dt=0.01):
        self._dt = dt
        self._g = 9.8  # Down gravity  (m/s^2)

        # [u; v; w; p; q; r; phi; theta; psi; x; y; h]
        n = 12 

        # [delta_e; delta_t; delta_a; delta_r]
        m = 4
        f = lambda x: np.array([[0.9226,	0.,	-6.3988,	0.,	-30.5251,	0.,	0.,	4.8922,	0.,	-0.0042,	0.,	-0.0961],
                                [0.,	-11897.7,	0.,	25656.2,	0.,	168165.2,	-156452.2,	0.,	609562.5,	0.,	667.4,	0.],
                                [0.3700,	0.,	-1.7213,	0.,	13.7310,	0.,	0.,	-1.1890,	0.,	-0.0011,	0.,	-0.0363],
                                [0.,	674.0,	0.,	-1455.7,	0.,	-9522.0,	8861.7,	0.,	-34529.3,	0.,	-37.8,	0.],
                                [-0.0002,	0.,	0.0106,	0.,	-0.1135,	0.,	0.,	-0.2402,	0.,	0.,	0.,	0.],
                                [0.,	69.8,	0.,	-151.0,	0.,	-987.5,	-918.7,	0.,	3578.9,	0.,	-3.9,	0.],
                                [0.,	-12.7,	0.,	27.8,	0.,	179.7,	-167.4,	0.,	650.1,	0.,	0.7,	0.],
                                [-0.0038,	0.,	0.0105,	0.,	0.8371,	0.,	0.,	-0.0880,	0.,	0.,	0.,	0.0003],
                                [0.,	0.1,	0.,	-0.2,	0.,	0.1,	0.9,	0.,	-3.6,	0.,	0.,	0.],
                                [-2.0837,	0.,	3.9301,	0.,	175.7865,	0.,	0.,	-9.4685,	0.,	0.0183,	0.,	0.4445],
                                [0.,	5.0,	0.,	-11.2,	0.,	-84.7,	78.8,	0.,	-249.2,	0.,	-0.4,	0.],
                                [0.4399,	0.,	-2.2627,	0.,	-12.9391,	0.,	0.,	53.7377,	0.,	-0.0023,	0.,	-0.0462],
                                ]) @ x

        g = lambda x: np.array([[70.6103,	-1.3542,	0.,	    0.],
                                [0.,	    0.,	    -1731.8,	5045.0],
                                [9.6435,	0.7383,	    0.,	    0.],
                                [0.,	    0.,	        96.7,	-285.6],
                                [-0.0907,	-0.0176,	0., 	0.],
                                [0.,	    0.,	        10.0,	-30.5],
                                [0.,	    0.,     	-1.7,	5],
                                [0.0467,	0.0343,	    0.,	    0.],
                                [0.,	    0.,	        0.,	    0.],
                                [-32.8769,	-27.12781,	0.,	    0.],
                                [0.,	    0.,	        0.,	    0.],
                                [10.8834,	-5.2662,	0.,	    0.]
                                ])
        super().__init__(n, m, f, g, dt)

    @property
    def state(self):
        return (self._rn, self._re, self._rd, self._psi)
    
    def __repr__(self):
        return f'Airplane'

class Static(Dynamics):
    '''
    '''
    def __init__(self, n, m):
        '''
        Args:
            n: dimension of state vector x 
            m: dimension of input vector u
        '''
        f = lambda x : np.zeros(n) @ x
        g = lambda x : np.zeros([n, m])
        super().__init__(n, m, f, g)

    def __repr__(self):
        return f'Static'

class CBFController:
    def __init__(self, initial_state, goal_h, dynamics, u_bounds, s=Simulation3d(), dt = 1e-4, n = 12, m = 4):
        self.s = s
        self.dynamics = dynamics
        self.u_bounds = u_bounds
        self.dt = dt
        self.n = n
        self.m = m
        self.initial_state = initial_state
        self.goal_state = np.zeros(len(initial_state), dtype=np.float16)
        self.goal_state[-1] = goal_h

        # stationary dynamics
        f = lambda x : np.zeros(self.n) # Drift function
        g = lambda u : np.zeros([self.n, self.m]) # Control function
        self.static_dyn = Dynamics(self.n, self.m, f, g)
        

    def add_specs(self, state):
        self.s.obsts = []
        self.s.cbf_by_agent=[]

        # values for each specification in SI Units, taken from UMichigan paper
        u_c = 1.3
        u_l = 2.6
        u_u = 5.1
        # u_l = 10.
        # u_u = 10.
        del_v = 1.51
        alpha = 3.
        alpha_rad = np.deg2rad(alpha)
        w_l = 0.
        w_u = 2.
        d_r = 3048.
        beta = 2.
        beta_rad = np.deg2rad(beta)
        d = 305.
        t = 305.
        alpha_h = 0.7
        alpha_h_rad = np.deg2rad(alpha_h)
        h_f = self.goal_state[-1]
        v_so = 45. #89.  # km/h (48 kcas)S  - specific to Cessna Skyhawk (not UMich value)

        ### spec inequalities
        bound = np.array([[0,0],[-10.,10.]]) # dummy parameter, doesn't actually mean anything
        #################### Long Spec 1: Velocity in x-dir u ####################
        spec1_geq = HalfPlaneND(n = np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), d = u_c*v_so-u_l, geq = 1, bound=bound)
        spec1_leq = HalfPlaneND(n = np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), d = u_c*v_so+u_u, geq = -1, bound=bound)
        
        o1_geq = Obstacle(state=spec1_geq.state, shape=spec1_geq, dynamics=self.static_dyn)
        o1_leq = Obstacle(state=spec1_leq.state, shape=spec1_leq, dynamics=self.static_dyn)
        # self.s.add_obstacle(obst=o1_geq)
        # self.s.add_obstacle(obst=o1_leq)

        ##################### Lat Spec 2: Lateral speed v ####################
        spec2_geq = HalfPlaneND(n = np.array([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), d = -del_v, geq = 1, bound=bound)
        spec2_leq = HalfPlaneND(n = np.array([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), d = del_v, geq = -1, bound=bound)
        
        o2_geq = Obstacle(state=spec2_geq.state, shape=spec2_geq, dynamics=self.static_dyn)
        o2_leq = Obstacle(state=spec2_leq.state, shape=spec2_leq, dynamics=self.static_dyn)
        # self.s.add_obstacle(obst=o2_geq)
        # self.s.add_obstacle(obst=o2_leq)
        
        #################### Long Spec 3: climb/descent rate w ####################
        spec3_geq = HalfPlaneND(n = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), d = w_l, geq = 1, bound=bound)
        spec3_leq = HalfPlaneND(n = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), d = state[0]*w_u*np.tan(alpha_rad), geq = -1, bound=bound)

        o3_geq = Obstacle(state=spec3_geq.state, shape=spec3_geq, dynamics=self.static_dyn)
        o3_leq = Obstacle(state=spec3_leq.state, shape=spec3_leq, dynamics=self.static_dyn)
        self.s.add_obstacle(obst=o3_geq)
        self.s.add_obstacle(obst=o3_leq)

        # #################### Lat Spec 4: Lateral deviation y ####################

        # spec4_geq = HalfPlaneND(n = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]), d = -(state[9]+d_r)*np.tan(beta_rad), geq = 1, bound=bound)
        # spec4_leq = HalfPlaneND(n = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]), d = (state[9]+d_r)*np.tan(beta_rad), geq = -1, bound=bound)
        # o4_geq = Obstacle(state=spec4_geq.state, shape=spec4_geq, dynamics=self.static_dyn)
        # o4_leq = Obstacle(state=spec4_leq.state, shape=spec4_leq, dynamics=self.static_dyn)

        # self.s.add_obstacle(obst=o4_geq)
        # self.s.add_obstacle(obst=o4_leq)

        # #################### Long Spec 5: vertical position h ####################
        # spec5_geq = HalfPlaneND(n = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]), d = (d-t)*np.tan(alpha_rad-alpha_h_rad), geq = 1, bound=bound)
        # spec5_leq = HalfPlaneND(n = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]), d = (d-t)*np.tan(alpha_rad+alpha_h_rad), geq = -1, bound=bound)
        
        # # half-plane 1 geq. Adding dynamics because spec is dependent on state x
        # f = lambda x : np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,np.tan(alpha_rad-alpha_h_rad),0.,0.]) # Drift function
        # g = lambda u : np.zeros([n,4]) # Control function
        # dynamics_spec5_geq = Dynamics(n,m,f,g)

        # # half-plane 2 leq. Adding dynamics because spec is dependent on state x
        # f = lambda x : np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,np.tan(alpha_rad+alpha_h_rad),0.,0.]) # Drift function
        # g = lambda u : np.zeros([n,4]) # Control function
        # dynamics_spec5_leq = Dynamics(n,m,f,g)

        # o5_geq = Obstacle(state=spec5_geq.state, shape=spec5_geq, dynamics=dynamics_spec5_geq)
        # o5_leq = Obstacle(state=spec5_leq.state, shape=spec5_leq, dynamics=dynamics_spec5_leq)
        # s.add_obstacle(obst=o5_geq)
        # s.add_obstacle(obst=o5_leq)

    def update_sim(self, u_ref, state):
        # u_ref is in x-plane sim control inputs. convert it to state-space controls

        for a in self.s.agents:
            # u_ref_bounds=[[-1,1],[0,1],[-1,1],[-1,1]]
            # controls = []
            # for u_idx in range(len(u_ref)):
            #     controls.append(np.interp(u_ref[u_idx], u_ref_bounds[u_idx], self.u_bounds[u_idx]))#+ u_prev[u_idx])
            # a.u_ref = self.xplane2ss(u_ref, u_prev)
            a.u_ref = u_ref
            a.goal = None
            a.state = state
        
        for idx in range(len(self.s.control)):
            control = self.s.control[idx]
            if isinstance(control,ReferenceControl):
                self.s.control[idx] = ReferenceControl(self.s.agents[0],np.array(self.s.agents[0].u_ref))
    
    def xplane2ss(self, u_ref, u_prev, u_ref_bounds=[[-1,1],[0,1],[-1,1],[-1,1]]):
        
        '''
        transform u_ref (x-plane controls) into state space controls

        u_prev : previous timesteps u given in state space controls
        u_ref_bounds bounds are the bounds given in x-plane controls
        '''
        u_transform = []
        for idx in range(len(u_ref)):
            # if u_ref == -998:
            #     u_transform.append(u_prev[idx])
            #     continue

            range_ref = u_ref_bounds[idx][1] -  u_ref_bounds[idx][0]
            range_sim = self.u_bounds[idx][1] -  self.u_bounds[idx][0]
            u_new = (((u_ref[idx] - u_ref_bounds[idx][0]) * range_sim) / range_ref) + self.u_bounds[idx][0]
            # u_new_prev = (((u_prev[idx] - u_ref_bounds[idx][0]) * range_sim) / range_ref) + self.u_bounds[idx][0]

            # if round(u_new - u_prev[idx],5) == 0:
            #     u_transform.append(-998)
            # else:

            # u_ref is relative -> make u_new absolute
            u_abs = np.clip(u_new+u_prev[idx], self.u_bounds[idx][0], self.u_bounds[idx][1])
            u_transform.append(u_new)

        return u_transform
    
    def setup(self, gamma, scaling=180., u_ref=False):
        a = Agent(self.initial_state, radius = 0.001, dynamics=self.dynamics)

        # use the provided goal point to create CLF, otherwise u_ref = reference control that will be updated in the main loop
        goal = Goal(self.goal_state, dynamics=self.static_dyn, gamma=gamma, idxs_ignore=[0,1,2,3,4,5,6,7,8,9,10])
        self.s.add_agent(agent=a, control=[0,0,0,0], lower_bounds=[lim[0] for lim in self.u_bounds], upper_bounds=[lim[1] for lim in self.u_bounds])
        self.add_specs(self.initial_state)

