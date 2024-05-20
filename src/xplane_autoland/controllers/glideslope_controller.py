import sys
sys.path.append('/home/agchadbo/XPlaneAutolandScenario/src/xplane_autoland')
from xplane_autoland.utils.pid import PID
import math

# TCH = Threshold Crossing Height
# Set one for Grant Co Intl Airport Runway 04
# 50 ft TCH -> meters
GRANT_RWY4_TCH = 50 * 0.3048

class GlideSlopeController:
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
            print("No height value")
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
