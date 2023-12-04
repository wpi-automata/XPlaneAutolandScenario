import math
import numpy as np
import time

from xplane_autoland.xplane_connect.xpc3 import XPlaneConnect

# Default values set for Rwy 04 of Grant Co. Airport
# TODOs relate to generalizing for arbitrary airports

class XPlaneDriver:
    def __init__(self, home_heading=53.7, local_start=(-35285.421875, 40957.0234375),
                 start_ground_range=12464, start_elev = 1029.45,
                 t=(-25159.26953, 33689.8125)):
        self._client = XPlaneConnect()
        self._home_heading = home_heading
        self._local_start = local_start
        # TODO: get from glideslope points
        self._start_ground_range = start_ground_range # (m)
        self._start_elev = start_elev # (m) beginning elevation for a 3 degree glideslope
        self._t = np.array(t).reshape((2, 1))

        # TODO: determine this automatically
        #       relates to home heading and local coordinate frame at given airport
        self._rotrad = -0.6224851011617226
        self._R = np.array([[ np.cos(self._rotrad), -np.sin(self._rotrad) ],
                            [ np.sin(self._rotrad),  np.cos(self._rotrad)]])

    def reset(self,
              init_u=60., init_v=0, init_w=0., init_p=0, init_q=0, init_r=0,
              init_phi=0, init_theta=0, init_psi=0, init_x=12464, init_y=0, init_h=None,
              noBrake=True):
        """
            Resets the aircraft and resets forces, fuel, etc.

            Args (all state variables):
                init_u        - longitudinal velocity (m/s)
                init_v        - lateral velocity (m/s)
                init_w        - vertical velocity (m/s)
                init_p        - roll rate (deg/s)
                init_q        - pitch rate (deg/s)
                init_r        - yaw rate (deg/s)
                init_phi      - roll angle (deg)
                init_theta    - pitch angle (deg)
                init_psi      - initial heading (deg)
                init_x        - horizontal distance (m)
                init_y        - lateral deviation (m)
                init_h        - aircraft altitude (m)
            Note: if choose not to pass one, will be set to a nominal default value
        """

        self._client.pauseSim(True)

        if init_h is None:
            init_h = self._start_elev

        # Turn off joystick "+" mark from screen
        self._client.sendDREF("sim/operation/override/override_joystick", 1)

        # Zero out control inputs
        self._client.sendCTRL([0,0,0,0])

        # Set parking brake
        self._client.sendDREF("sim/flightmodel/controls/parkbrake", int(noBrake))

        # Zero out moments and forces
        initRef = "sim/flightmodel/position/"
        drefs = []
        refs = ['theta','phi', 'local_vx','local_vy','local_vz','local_ax','local_ay','local_az',
        'Prad','Qrad','Rrad','q','groundspeed',
        'indicated_airspeed','indicated_airspeed2','true_airspeed','M','N','L','P','Q','R','P_dot',
        'Q_dot','R_dot','Prad','Qrad','Rrad']
        for ref in refs:
            drefs += [initRef+ref]
        values = [0]*len(refs)
        self._client.sendDREFs(drefs,values)

        # Set position and orientation
        # Set known good start values
        # Note: setting position with lat/lon gets you within 0.3m. Setting local_x, local_z is more accurate)
        self.set_orient_pos(init_phi, init_theta, init_psi, init_x, init_y, init_h)
        self.set_orientrate_vel(init_u, init_v, init_w, init_p, init_q, init_r)

        # Fix the plane if you "crashed" or broke something
        self._client.sendDREFs(["sim/operation/fix_all_systems"], [1])

        # Set fuel mixture for engine
        self._client.sendDREF("sim/flightmodel/engine/ENGN_mixt", 0.61)

        # Reset fuel levels
        self._client.sendDREFs(["sim/flightmodel/weight/m_fuel1","sim/flightmodel/weight/m_fuel2"],[232,232])

        # Give time to settle
        time.sleep(1.)

    def pause(self, yes=True):
        """
        Pause or unpause the simulation

        Args:
            yes: whether to pause or unpause the sim [default: True (i.e., pause)]
        """
        self._client.pauseSim(yes)

    def send_ctrl(self, elev, aileron, rudder, throttle):
        """
        Sets control surface information (on the main aircraft)

            Args:
                elev: [-1, 1]
                aileron: [-1, 1]
                rudder: [-1, 1]
                throttle: [-1, 1]
        """
        self._client.sendCTRL([elev, aileron, rudder, throttle])

    def send_brake(self, brake):
        """
        Set the parking brake to on or off

            Args:
                brake: 0 - off; 1 - on
        """
        self._client.sendDREF("sim/flightmodel/controls/parkbrake", brake)

    ###########################################################################
    # State estimation
    # by default these are passthroughs -- can be overloaded in a subclass
    ###########################################################################
    def est_statevec(self, img):
        vel  = self.est_vel_state()
        ovel = self.est_orient_vel_state()
        o    = self.est_orient_state()
        pos  = self.est_pos_state()

        return np.stack((vel, ovel, o, pos)).flatten()

    def est_vel_state(self):
        return self.get_vel_state()

    def est_orient_vel_state(self):
        return self.get_orient_vel_state()

    def est_orient_state(self):
        return self.get_orient_state()

    def est_pos_state(self, img):
        return self.get_pos_state()

    ###########################################################################
    # True state getters
    ###########################################################################
    def get_statevec(self):
        """
        Returns the state vector used in the autoland scenario
        Based on https://arc.aiaa.org/doi/10.2514/6.2021-0998
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
        """

        vel  = self.get_vel_state()
        ovel = self.get_orient_vel_state()
        o    = self.get_orient_state()
        pos  = self.get_pos_state()

        return np.stack((vel, ovel, o, pos)).flatten()

    def get_vel_state(self):
        return self._body_frame_velocity()

    def get_orient_vel_state(self):
        P = self._client.getDREF('sim/flightmodel/position/P')[0]
        Q = self._client.getDREF('sim/flightmodel/position/Q')[0]
        R = self._client.getDREF('sim/flightmodel/position/R')[0]
        return np.array([P, Q, R])

    def get_orient_state(self):
        phi = self._client.getDREF('sim/flightmodel/position/phi')[0]
        theta = self._client.getDREF('sim/flightmodel/position/theta')[0]
        psi = self._get_home_heading()
        return np.array([phi, theta, psi])

    def get_pos_state(self):
        x, y = self._get_home_xy()
        h = self._client.getDREF('sim/flightmodel/position/elevation')[0]
        return np.array([x, y, h])

    def set_orient_pos(self, phi, theta, psi, x, y, h):
        '''
        Set the orientation and position of the plane.
        Args:
            phi    - roll angle (deg)
            theta  - pitch angle (deg)
            psi    - yaw angle (deg)
            x      - horizontal distance (m)
            y      - lateral deviation (m)
            h      - aircraft altitude (m)
        '''
        # zero out orientation at first
        self._client.sendDREF('sim/flightmodel/position/phi', 0)
        self._client.sendDREF('sim/flightmodel/position/theta', 0)
        self._client.sendDREF('sim/flightmodel/position/psi', self._to_local_heading(0))

        self._send_xy(x, y)
        # set elevation by getting offset between local y (the axis for elevation)
        # and current elevation
        # then use that to shift the coordinate to align with the desired elevation
        curr_elev = self._client.getDREF("sim/flightmodel/position/elevation")[0]
        curr_localy = self._client.getDREF("sim/flightmodel/position/local_y")[0]
        offset = curr_elev - curr_localy
        self._client.sendDREF("sim/flightmodel/position/local_y", h - offset)

        self._client.sendDREF('sim/flightmodel/position/phi', phi)
        self._client.sendDREF('sim/flightmodel/position/theta', theta)
        self._client.sendDREF('sim/flightmodel/position/psi', self._to_local_heading(psi))

    def set_orientrate_vel(self, u, v, w, p, q, r):
        # 2d rotation and flip
        uv = np.array([u, v]).reshape((2, 1))
        hr = math.radians(self._home_heading)
        R = np.array([[np.cos(hr), -np.sin(hr)],[np.sin(hr), np.cos(hr)]])
        rot_uv = R@uv
        # flip direction of longitudinal velocity to put in OpenGL coordinates
        self._client.sendDREF('sim/flightmodel/position/local_vz', -rot_uv[0])
        self._client.sendDREF('sim/flightmodel/position/local_vx', rot_uv[1])
        self._client.sendDREF('sim/flightmodel/position/local_vy', w)
        self._client.sendDREF('sim/flightmodel/position/P', p)
        self._client.sendDREF('sim/flightmodel/position/Q', q)
        self._client.sendDREF('sim/flightmodel/position/R', r)
    
    ###########################################################################
    # Helper functions
    ###########################################################################
    def _body_frame_velocity(self):
        cos = math.cos
        sin = math.sin

        psi = self._client.getDREF('sim/flightmodel/position/psi')[0]
        theta = self._client.getDREF('sim/flightmodel/position/theta')[0]
        phi = self._client.getDREF('sim/flightmodel/position/phi')[0]

        h = math.radians(psi)
        Rh = np.array([[ cos(h), sin(h), 0],
                    [-sin(h), cos(h), 0],
                    [      0,      0,  1]])
        el = math.radians(theta)
        Re = np.array([[cos(el), 0, -sin(el)],
                    [      0, 1,        0],
                    [sin(el),  0,  cos(el)]])
        roll = math.radians(phi)
        Rr = np.array([[1,          0,         0],
                    [0,  cos(roll), sin(roll)],
                    [0, -sin(roll), cos(roll)]])
        R = np.matmul(Rr, np.matmul(Re, Rh))

        vx = self._client.getDREF('sim/flightmodel/position/local_vx')[0]
        vy = self._client.getDREF('sim/flightmodel/position/local_vy')[0]
        vz = self._client.getDREF('sim/flightmodel/position/local_vz')[0]
        # local frame is East-Up-South and we convert to North-East-Down
        vel_vec = np.array([-vz, vx, -vy]).T

        return np.matmul(R, vel_vec)

    def _get_home_heading(self):
        """
        Get the value of the aircraft's heading in degrees from the runway
        """
        true_heading = self._client.getDREF("sim/flightmodel/position/psi")[0]
        return true_heading - self._home_heading

    def _get_local_heading(self):
        """
        Get the value of the aircraft's heading in degrees from the Z axis
        """
        return self._client.getDREF("sim/flightmodel/position/psi")[0]

    def _get_home_xy(self):
        """
        Get the aircraft's current x and y position and heading in the
        home frame. The x-value represents crosstrack error,the y-value represents
        downtrack position, and theta is the heading error.
        """

        psi = self._client.getDREF("sim/flightmodel/position/psi")[0]
        x   = self._client.getDREF("sim/flightmodel/position/local_x")[0]
        y   = self._client.getDREF("sim/flightmodel/position/local_z")[0]

        # Get the positions in home coordinates
        rotx, roty = self._local_to_home(x, y)

        x = self._start_ground_range - roty
        y = -rotx
        return x, y

    # TODO: replace with single-step coordinate transform
    def rotateToLocal(self, x, y):
        """Rotate to the local coordinate frame.

            Args:
                x: x-value in home coordinate frame
                y: y-value in home coordinate frame
        """
        rotx = 0.583055934597441 * x + -0.8124320138514389 * y
        roty = 0.8124320138514389 * x + 0.583055934597441 * y
        return rotx, roty

    # TODO: replace with single-step coordinate transformation
    def _local_to_home(self, x, y):
        """Get the home coordinates of the aircraft from the local coordinates.

            Args:
                x: x-value in the local coordinate frame
                y: y-value in the local coordinate frame
        """

        # Translate to make start x and y the origin
        startX, startY = self._local_start
        transx = startX - x
        transy = startY - y

        # Rotate to align runway with y axis
        rotx, roty = self.rotate_to_home(transx, transy)
        return rotx, roty

    def rotate_to_home(self, x, y):
        """
            Rotate to the home coordinate frame.

            Home coordinate frame starts at (0,0) at the start of the runway
            and ends at (0, 2982 at the end of the runway). Thus, the x-value
            in the home coordinate frame corresponds to crosstrack error and
            the y-value corresponds to downtrack position.

            Args:
                x: x-value in local coordinate frame
                y: y-value in local coordinate frame
        """
        rotx = 0.583055934597441 * x + 0.8124320138514389 * y
        roty = -0.8124320138514389 * x + 0.583055934597441 * y
        return rotx, roty

    def _to_local_heading(self, psi):
        """
        Convert home heading to local frame heading
        """
        return psi + self._home_heading

    def _send_xy(self, x, y):
        local_x, local_z = self._xy_to_local_xz(x, y)
        self._client.sendDREF("sim/flightmodel/position/local_x", local_x)
        self._client.sendDREF("sim/flightmodel/position/local_z", local_z)

    def _xy_to_local_xz(self, x, y):
        """
        Converts autoland statevec's x, y elements to local x, z coordinates.
        Note: in local frame, y is elevation (up) so we care about x and **z** for this rotation
        """
        # rotation to align to runway
        # flip a sign because of x, y orientation
        # in autoland frame, x is pointing frame the runway to the starting point
        # and y is pointing to the right from the plane's point of view
        F = np.array([[-1.,  0.],
                    [ 0., 1.]])
        r = (self._R@F)@np.array([[x], [y]]).reshape((2, 1))
        local_x, local_z = r + self._t
        return local_x.flatten(), local_z.flatten()

    def _local_xz_to_xy(self, local_x, local_z):
        R = self._R
        F = np.array([[-1.,  0.],
                    [ 0., 1.]])
        RF = R@F
        l = np.array([[local_x], [local_z]]).reshape((2, 1))
        r = l - self._t
        x, y = np.linalg.inv(RF)@r
        return x, y

