import math
import numpy as np
import time

from xplane_autoland.xplane_connect.xpc3 import XPlaneConnect

class XPlaneDriver:
    def __init__(self, home_heading=53.7, local_start=(-35285.421875, 40957.0234375),
                 start_elev = 1029.45):
        self._client = XPlaneConnect()
        self._home_heading = home_heading
        self._local_start = local_start
        # TODO: get from glideslope points
        self._start_elev = start_elev # beginning elevation for a 3 degree glideslope


    def reset(self, cteInit=0, heInit=0, dtpInit=0, noBrake=True):
        """
            Resets the aircraft and resets forces, fuel, etc.

            Args:
                cteInit: initial crosstrack error (meters)
                heInit: initial heading error (degrees)
                dtpInit: initial downtrack position (meters)
        """

        self._client.pauseSim(True)

        # Turn off joystick "+" mark from screen
        self._client.sendDREF("sim/operation/override/override_joystick", 1)

        # Zero out control inputs
        self._client.sendCTRL([0,0,0,0])

        # Set parking brake
        self._client.sendDREF("sim/flightmodel/controls/parkbrake", int(noBrake))

        # Zero out moments and forces
        initRef = "sim/flightmodel/position/"
        drefs = []
        refs = ['theta','phi','psi','local_vx','local_vy','local_vz','local_ax','local_ay','local_az',
        'Prad','Qrad','Rrad','q','groundspeed',
        'indicated_airspeed','indicated_airspeed2','true_airspeed','M','N','L','P','Q','R','P_dot',
        'Q_dot','R_dot','Prad','Qrad','Rrad']
        for ref in refs:
            drefs += [initRef+ref]
        values = [0]*len(refs)
        self._client.sendDREFs(drefs,values)

        # Set position and orientation
        # Set known good start values
        # TODO: remove this and get start position based on runway start
        self._client.sendPOSI([47.196890, -119.33260, 362.14444, 0.31789625, 0.10021035, 53.7, 1], 0)
        # Fine-tune position
        # Setting position with lat/lon gets you within 0.3m. Setting local_x, local_z is more accurate)
        self.setHomeState(cteInit, dtpInit, heInit)

        # Fix the plane if you "crashed" or broke something
        self._client.sendDREFs(["sim/operation/fix_all_systems"], [1])

        # Set fuel mixture for engine
        self._client.sendDREF("sim/flightmodel/engine/ENGN_mixt", 0.61)

        # Set speed of aircraft to be 60 m/s in current heading direction
        heading = self._home_heading - heInit
        self._client.sendDREF("sim/flightmodel/position/local_vx", 60.0*np.sin(heading*np.pi/180.0))
        self._client.sendDREF("sim/flightmodel/position/local_vz", -60.0*np.cos(heading*np.pi/180.0))

        # Reset fuel levels
        self._client.sendDREFs(["sim/flightmodel/weight/m_fuel1","sim/flightmodel/weight/m_fuel2"],[232,232])

        ###########################################################################
        # State estimation
        # by default these are passthroughs -- can be overloaded in a subclass
        ###########################################################################
        def est_statevec(self):
            return self.get_statevec()

        def est_vel_state(self):
            return self.get_vel_state()

        def est_orient_vel_state(self):
            return self.get_orient_vel_state()

        def est_orient_state(self):
            return self.get_orient_state()

        def est_pos_state(self):
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

        vel = self._body_frame_velocity()

        P = self._client.getDREF('sim/flightmodel/position/P')[0]
        Q = self._client.getDREF('sim/flightmodel/position/Q')[0]
        R = self._client.getDREF('sim/flightmodel/position/R')[0]

        phi = self._client.getDREF('sim/flightmodel/position/phi')[0]
        theta = self._client.getDREF('sim/flightmodel/position/theta')[0]
        psi = self._get_home_heading()

        # runway distances (different frame than home)
        x, y = self._get_home_xy()
        h = self._client.getDREF('sim/flightmodel/position/elevation')[0]

        return np.array([
            vel[0],
            vel[1],
            vel[2],
            P,
            Q,
            R,
            phi,
            theta,
            psi,
            x,
            y,
            h
        ]).T

    def get_vel_state(self):
        pass

    def get_orient_vel_state(self):
        pass

    def get_orient_state(self):
        pass

    def get_pos_state(self):
        pass

    
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

        x = self._get_autoland_runway_thresh() - roty
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


    # TODO: replace with single-step coordinate transform
    def homeToLocal(self, x, y):
        """Get the local coordinates of the aircraft from the home coordinates.

            Args:
                x: x-value in the home coordinate frame
                y: y-value in the home coordinate frame
        """

        # Rotate back
        rotx, roty = self.rotateToLocal(x, y)

        # Translate back
        startX, startY = self._local_start
        transx = startX - rotx
        transy = startY - roty

        return transx, transy


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

    # TODO: replace this (don't have three coordinate systems)
    def _get_autoland_runway_thresh(self):
        '''
        The runway threshold along the y-axis in the home frame
        '''
        return 12464

    # TODO: replace this
    def setHomeState(self, x, y, theta):
        """Set the aircraft's state using coordinates in the home frame.
        This is equivalent to setting the crosstrack error (x), downtrack
        position (y), and heading error (theta).

            Args:
                x: desired crosstrack error [-10, 10] (meters)
                y: desired downtrack position [0, 2982] (meters)
                theta: desired heading error [-180, 180] (degrees)
        """

        localx, localz = self.homeToLocal(x, y)

        self._client.sendDREF("sim/flightmodel/position/local_x", localx)
        self._client.sendDREF("sim/flightmodel/position/local_z", localz)
        self._client.sendDREF("sim/flightmodel/position/psi", self._home_heading - theta)

        time.sleep(0.02)

        # TODO: make this configurable
        startElev = 1029.45 # beginning elevation for a 3 degree glideslope
        curr_elev = self._client.getDREF("sim/flightmodel/position/elevation")[0]
        curr_localy = self._client.getDREF("sim/flightmodel/position/local_y")[0]
        offset = curr_elev - curr_localy
        self._client.sendDREF("sim/flightmodel/position/local_y", startElev - offset)