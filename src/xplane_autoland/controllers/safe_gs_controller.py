from xplane_autoland.utils.pid import PID
import math

# TCH = Threshold Crossing Height
# Set one for Grant Co Intl Airport Runway 04
# 50 ft TCH -> meters
GRANT_RWY4_TCH = 50 * 0.3048

class ydot_controller:
    def __init__(self, gamma, tch=GRANT_RWY4_TCH, runway_elev=361, des_u=50., dt=0.1):
        self._gamma       = gamma    # glide slope angle
        self._h_thresh    = tch + runway_elev # height of runway threshold (m) is the TCH + the elevation of the runway (m)
        self._runway_elev = runway_elev
        self._des_u       = des_u # desired longitudinal velocity (m/s)
        self._dt          = dt

        if dt > 0.5:
            raise Warning("Running at a much slower dt than controller was designed for")

        self._tan_gamma = math.tan(math.radians(self._gamma))

        self.prev_y = None

        # PI controllers
        # lateral
        self._ydot_pid  = PID(dt, kp=0.5, ki=0.0, kd=4.)
        self._phi_pid   = PID(dt, kp=1., ki=0.1, kd=0.)

        # vertical
        self._u_pid     = PID(dt, kp=50., ki=5., kd=0.)
        self._theta_pid = PID(dt, kp=0.24, ki=0.024, kd=0.)

    @property
    def runway_elevation(self):
        '''
        Returns the target height above the runway threshold
        '''
        return self._runway_elev
    
    def get_y_dot(self, statevec):
        """
        Returns component of velocity lateral to the runway
        """
        u,v = statevec[0:2]
        psi = statevec[8]
        ydot = u*math.sin(math.radians(psi)) + v*math.cos(math.radians(psi))
        return ydot

    def desired_ydot(self, statevec):
        '''Standin calculation for a ydot to zero y'''
        y = statevec[10]
        return 0.

    def control(self, statevec, err_h=None):
        u, v, w, \
        p, q, r, \
        phi, theta, psi, \
        x, y, h = statevec

        if self.prev_y is None:
            self.prev_y = y

        err_phi = 0.0 - phi

        # lateral control
        ydot = self.get_y_dot(statevec)
        err_ydot = self.desired_ydot(statevec) - ydot

        delta_r = self._ydot_pid(err_ydot)
        delta_a = self._phi_pid(err_phi)

        rudder = max(-27, min(delta_r, 27))/27
        aileron = max(-20, min(delta_a, 20))/20

        # longitudinal control
        err_u = self._des_u - u

        fu = self._u_pid(err_u) + 5000
        throttle = min(fu, 10000)/10000

        if err_h is None:
            h_c = self.get_glideslope_height_at(x)
            err_h = h_c - h
        theta_c = self._theta_pid(err_h)
        elev = (theta_c - theta)*5 - 0.05*q

        if elev > 0:
            elevator = min(elev, 30)/30
        else:
            elevator = max(elev, -15)/15

        return elevator, aileron, rudder, throttle

    @property
    def runway_threshold_height(self):
        return self._h_thresh

    def get_glideslope_height_at(self, x):
        return self._h_thresh + x*self._tan_gamma


class SafeGlideSlopeController:
    def __init__(self, gamma, tch=GRANT_RWY4_TCH, runway_elev=361, des_u=50., dt=0.1):
        self._gamma       = gamma    # glide slope angle
        self._h_thresh    = tch + runway_elev # height of runway threshold (m) is the TCH + the elevation of the runway (m)
        self._runway_elev = runway_elev
        self._des_u       = des_u # desired longitudinal velocity (m/s)
        self._dt          = dt

        if dt > 0.5:
            raise Warning("Running at a much slower dt than controller was designed for")

        self._tan_gamma = math.tan(math.radians(self._gamma))

        # PI controllers
        # lateral
        self._psi_pid   = PID(dt, kp=1., ki=0.1, kd=0.)
        self._y_pid     = PID(dt, kp=0.5, ki=0.01, kd=0.)
        self._phi_pid   = PID(dt, kp=1., ki=0.1, kd=0.)

        self._u_pid     = PID(dt, kp=50., ki=5., kd=0.)
        self._theta_pid = PID(dt, kp=0.24, ki=0.024, kd=0.)

    @property
    def runway_elevation(self):
        '''
        Returns the target height above the runway threshold
        '''
        return self._runway_elev

    def control(self, statevec, err_h=None):
        '''
        INPUTS
            Statevector based on https://arc.aiaa.org/doi/10.2514/6.2021-0998
            statevec with components
                u      - longitudinal velocity (m/s)
                v      - lateral velocity (m/s)
                w      - vertical velocity (m/s)
                p      - roll velocity (deg/s)
                q      - pitch velocity (deg/s)
                r      - yaw velocity (deg/s)
                phi    - roll angle (deg)
                theta  - pitch angle (deg)
                psi    - yaw angle (deg)
                x      - horizontal distance (m)
                y      - lateral deviation (m)
                h      - aircraft altitude (m)
            err_h
                The error in height
                If not provided, it will be calculated based on glideslope angle gamma, x, and h
                This option is for the vision network which directly predicts height error rather than runway distance x
        OUTPUTS
            throttle
            elevator
            rudder
            aileron
        '''

        u, v, w, \
        p, q, r, \
        phi, theta, psi, \
        x, y, h = statevec

        # lateral control
        err_y = 0.0 - y
        err_psi = 0.0 - psi
        err_phi = 0.0 - phi

        delta_r = self._psi_pid(err_psi) + self._y_pid(err_y)
        delta_a = self._phi_pid(err_phi)
        rudder = max(-27, min(delta_r, 27))/27
        aileron = max(-20, min(delta_a, 20))/20

        # longitudinal control
        err_u = self._des_u - u

        fu = self._u_pid(err_u) + 5000
        throttle = min(fu, 10000)/10000

        if err_h is None:
            h_c = self.get_glideslope_height_at(x)
            err_h = h_c - h
        theta_c = self._theta_pid(err_h)
        elev = (theta_c - theta)*5 - 0.05*q

        if elev > 0:
            elevator = min(elev, 30)/30
        else:
            elevator = max(elev, -15)/15

        return elevator, aileron, rudder, throttle

    @property
    def runway_threshold_height(self):
        return self._h_thresh

    def get_glideslope_height_at(self, x):
        return self._h_thresh + x*self._tan_gamma

if __name__ == '__main__':
    import numpy as np
    from xplane_autoland.controllers.plane_model import PlaneModel
    import matplotlib.pyplot as plt

    start_elev = 12464
    start_ground_range=12464
    
    dt = 0.1
    max_time = 100 #600
    gsc = SafeGlideSlopeController(gamma=3, dt=dt)
    # the altitude of the runway threshold
    h_thresh = gsc.runway_threshold_height
    
    slope = float(start_elev - h_thresh) / start_ground_range

    # distance from the runway crossing (decrease to start closer)
    # vision mode works best at 9000m and less (up until right before landing)
    # need to train more at higher distances and close up
    x_val = 9000
    init_h = slope * x_val + h_thresh
    # can set the state arbitrarily (see reset documentation for semantics)

    init_state = np.array([ 6.00000002e+01,  1.19646966e-07,  0.00000000e+00,  0.00000000e+00,
                            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                            7.62939450e-07,  8.99698899e+03, -3.31549693e-02,  8.47875427e+02])
    plane = PlaneModel(init_state)
    
    x_hist = np.ndarray((12, math.ceil(max_time/dt)))
    for step in range(math.ceil(max_time/dt)):
        state = plane.state
        phi, theta, psi, x, y, h = state[-6:]
        err_h = None

        elevator, aileron, rudder, throttle = gsc.control(state, err_h=err_h)        
        # Step the model forward
        plane.send_ctrl(elevator, aileron, rudder, throttle)
        x_hist[:,step] = plane.state
    
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(0,max_time,dt), x_hist[10,:].T)
    plt.show()
        
