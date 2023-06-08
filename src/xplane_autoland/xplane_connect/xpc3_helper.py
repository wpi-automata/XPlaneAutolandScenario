# Helper functions for xpc3 specific to the taxinet problem
# Written by Kyle Julian and Sydney Katz
# Maintained by Sydney Katz (smkatz@stanford.edu)
# Updated by Makai Mann (makai.mann@ll.mit.edu)

import xplane_autoland.xplane_connect.xpc3

import numpy as np
import math
import time
import pandas as pd
from scipy.spatial.transform import Rotation

# Heading to align with the runway
HOME_HEADING = 53.7

def sendCTRL(client, elev, aileron, rudder, throttle):
    """Sets control surface information (on the main aircraft)

        Args:
            client: XPlane Client
            elev: [-1, 1]
            aileron: [-1, 1]
            rudder: [-1, 1]
            throttle: [-1, 1]
    """
    client.sendCTRL([elev, aileron, rudder, throttle])

def sendBrake(client, brake):
    """Set the parking brake to on or off

        Args:
            client: XPlane Client
            brake: 0 - off; 1 - on
    """
    client.sendDREF("sim/flightmodel/controls/parkbrake", brake)

def getSpeed(client):
    """Get current speed (m/s)

        Args:
            client: XPlane Client
    """
    return client.getDREF("sim/flightmodel/position/groundspeed")[0]


def getBrake(client):
    """Get current brake

        Args:
            client: XPlane Client
    """
    return client.getDREF("sim/flightmodel/controls/parkbrake")[0]

"""Position on Runway"""

def getStartXY(mode='land'):
    """Get Start x and y values in the local coordinate frame"""
    if mode == 'taxi':
        return -25159.26953125, 33689.8125
    elif mode == 'land':
        # with this as the origin,
        # runway crossing is at y~=12464
        return -35285.421875, 40957.0234375
    else:
        raise ValueError('Unexpected mode {mode} for getting start position')

def rotateToHome(x, y):
    """Rotate to the home coordinate frame.

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

def rotateToLocal(x, y):
    """Rotate to the local coordinate frame.

        Args:
            x: x-value in home coordinate frame
            y: y-value in home coordinate frame
    """
    rotx = 0.583055934597441 * x + -0.8124320138514389 * y
    roty = 0.8124320138514389 * x + 0.583055934597441 * y
    return rotx, roty

def homeToLocal(x, y):
    """Get the local coordinates of the aircraft from the home coordinates.

        Args:
            x: x-value in the home coordinate frame
            y: y-value in the home coordinate frame
    """

    # Rotate back
    rotx, roty = rotateToLocal(x, y)

    # Translate back
    startX, startY = getStartXY()
    transx = startX - rotx
    transy = startY - roty

    return transx, transy

def localToHome(x, y):
    """Get the home coordinates of the aircraft from the local coordinates.

        Args:
            x: x-value in the local coordinate frame
            y: y-value in the local coordinate frame
    """

    # Translate to make start x and y the origin
    startX, startY = getStartXY()
    transx = startX - x
    transy = startY - y

    # Rotate to align runway with y axis
    rotx, roty = rotateToHome(transx, transy)
    return rotx, roty

def getHomeState(client):
    """Get the aircraft's current x and y position and heading in the
    home frame. The x-value represents crosstrack error,the y-value represents
    downtrack position, and theta is the heading error.

        Args:
            client: XPlane Client
    """

    psi = client.getDREF("sim/flightmodel/position/psi")[0]
    x = client.getDREF("sim/flightmodel/position/local_x")[0]
    y = client.getDREF("sim/flightmodel/position/local_z")[0]

    # Rotate heading into home coordinates
    theta = HOME_HEADING - psi

    # Get the positions in home coordinates
    rotx, roty = localToHome(x, y)

    return rotx, roty, theta

def setHomeState(client, x, y, theta):
    """Set the aircraft's state using coordinates in the home frame.
    This is equivalent to setting the crosstrack error (x), downtrack
    position (y), and heading error (theta).

        Args:
            client: XPlane Client
            x: desired crosstrack error [-10, 10] (meters)
            y: desired downtrack position [0, 2982] (meters)
            theta: desired heading error [-180, 180] (degrees)
    """

    localx, localz = homeToLocal(x, y)

    client.sendDREF("sim/flightmodel/position/local_x", localx)
    client.sendDREF("sim/flightmodel/position/local_z", localz)
    client.sendDREF("sim/flightmodel/position/psi", HOME_HEADING - theta)

    # FIXME - reset doesn't work repeatedly because curr_agly doesn't get it from the ground
    # Place perfectly on the ground
    # Pause for a bit for it to move
    time.sleep(0.02)
    # startAGL = 1005.84 - 342.31 # m; initial elevation - ground elevation (converting to AGL)
    # curr_agly = client.getDREF("sim/flightmodel/position/y_agl")[0]
    # curr_localy = client.getDREF("sim/flightmodel/position/local_y")[0]
    # client.sendDREF("sim/flightmodel/position/local_y",
    #                 curr_localy - curr_agly + startAGL)

    startElev = 1029.45 # beginning elevation for a 3 degree glideslope
    curr_elev = client.getDREF("sim/flightmodel/position/elevation")[0]
    curr_localy = client.getDREF("sim/flightmodel/position/local_y")[0]
    offset = curr_elev - curr_localy
    client.sendDREF("sim/flightmodel/position/local_y", startElev - offset)

def getPercDownRunway(client):
    """Get the percent down the runway of the main aircraft

        Args:
            client: XPlane Client
    """

    x = client.getDREF("sim/flightmodel/position/local_x")[0]
    y = client.getDREF("sim/flightmodel/position/local_z")[0]
    _, yhome = localToHome(x, y)
    return (yhome / 2982) * 100

"""Utility"""

def reset(client, cteInit=0, heInit=0, dtpInit=0, noBrake=True):
    """Resets the aircraft and resets forces, fuel, etc.

        Args:
            client: XPlane Client
            cteInit: initial crosstrack error (meters)
            heInit: initial heading error (degrees)
            dtpInit: initial downtrack position (meters)
    """

    client.pauseSim(True)
    # lat,lon, alt, pitch, roll, heading = getStartPosition(cteInit, heInit, drInit);

    # Turn off joystick "+" mark from screen
    client.sendDREF("sim/operation/override/override_joystick", 1)

    # Zero out control inputs
    sendCTRL(client,0,0,0,0)

    # Set parking brake
    if noBrake:
        sendBrake(client,0)
    else:
        sendBrake(client,1)

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
    client.sendDREFs(drefs,values)

    # Set position and orientation
    # Set known good start values
    client.sendPOSI([47.196890, -119.33260, 362.14444, 0.31789625, 0.10021035, 53.7, 1], 0)
    # Fine-tune position
    # Setting position with lat/lon gets you within 0.3m. Setting local_x, local_z is more accurate)
    setHomeState(client, cteInit, dtpInit, heInit)

    # Fix the plane if you "crashed" or broke something
    client.sendDREFs(["sim/operation/fix_all_systems"], [1])

    # Set fuel mixture for engine
    client.sendDREF("sim/flightmodel/engine/ENGN_mixt", 0.61)

    # Set speed of aircraft to be 60 m/s in current heading direction
    heading = 53.7 - heInit
    client.sendDREF("sim/flightmodel/position/local_vx", 60.0*np.sin(heading*np.pi/180.0))
    client.sendDREF("sim/flightmodel/position/local_vz", -60.0*np.cos(heading*np.pi/180.0))

    # Reset fuel levels
    client.sendDREFs(["sim/flightmodel/weight/m_fuel1","sim/flightmodel/weight/m_fuel2"],[232,232])

def saveState(client, folder, filename='test.csv'):
    """Save the current state of the simulator to a CSV file.

        Pulls all relevant DREFs and stores them in a CSV.
        Can use loadState to set simulator back to a saved state.

        Args:
            client: XPlane Client
            folder: path to save to
            filename: name of file to save to
    """

    # Position/orientation/speeds/turn rates/etc datarefs
    drefs = []
    initRef = "sim/flightmodel/position/"
    refs = ['theta','phi','psi','local_x','local_y','local_z','local_vx','local_vy','local_vz','local_ax','local_ay',
    'local_az','Prad','Qrad','Rrad','q','groundspeed'
    'indicated_airspeed','indicated_airspeed2','M','N','L','P','Q','R','P_dot',
    'Q_dot','R_dot','Prad','Qrad','Rrad']
    for ref in refs:
        drefs += [initRef+ref]

    # Engine related datarefs
    initRef = "sim/flightmodel/engine/"
    refs = ['ENGN_N2_','ENGN_N1_','ENGN_EGT','ENGN_ITT','ENGN_CHT','ENGN_EGT_c','ENGN_ITT_c',
            'ENGN_CHT_C','ENGN_FF_','ENGN_EPR','ENGN_MPR','ENGN_oil_press_psi',
            'ENGN_oil_press','ENGN_oil_press','ENGN_power','ENGN_prop','ENGN_TRQ',
            'ENGN_thro','ENGN_thro_use',
            'POINT_thrust','POINT_tacrad','ENGN_mixt','ENGN_prop','ENGN_propmode']
    for ref in refs:
        drefs += [initRef+ref]

    # Force related dataref
    initRef = "sim/flightmodel/forces/"
    refs = ['fside_prop','fnrml_prop','faxil_prop','L_prop','M_prop','N_prop',
            'L_total','M_total','N_total']
    for ref in refs:
        drefs += [initRef+ref]

    # parking brake, time of day, and fuel level data ref
    drefs += ["sim/flightmodel/controls/parkbrake"]
    drefs += ["sim/time/zulu_time_sec"]
    drefs += ["sim/flightmodel/weight/m_fuel1","sim/flightmodel/weight/m_fuel2"]

    # Get the datarefs
    values = client.getDREFs(drefs)
    valuesFilt = []
    drefsFilt = []
    for i,val in enumerate(values):
        if len(val)>0:
            valuesFilt += [val[0]]
            drefsFilt += [drefs[i]]

    # Get position and controller settings
    valuesFilt += client.getPOSI()
    valuesFilt += client.getCTRL()
    drefsFilt += ["lat","lon","alt","pitch","roll","heading","gear"]
    drefsFilt += ["elev","aileron","rudder","throttle","gear","flaps","speedbrakes"]
    values = np.array(valuesFilt).reshape((1,len(valuesFilt)))

    # Save to CSV
    outData = pd.DataFrame(values,index=[0],columns=drefsFilt)
    csv_file = folder + "/"+filename
    outData.to_csv(csv_file,index=False,index_label=False)


def loadState(client, folder, filename='test.csv'):
    """Read csv file of saved simulator state, load the simulator with saved values

        Args:
            client: XPlane Client
            folder: path to file with the state
            filename: name of file with the state
    """

    # Read CSV file
    tab = pd.read_csv(folder + "/" +filename)
    drefs = list(tab.columns)
    values = list(tab.values[0])

    # Separate out values
    pos = values[-14:-7]
    ctrl = values[-7:]
    values = values[:-14]
    drefs = drefs[:-14]

    # Send values to simulator
    client.sendPOSI(pos)
    client.sendDREFs(drefs,values)
    ctrl[4] = int(ctrl[4]); ctrl[5] = int(ctrl[5]); ctrl[6] = int(ctrl[6]);
    ctrl[3]*=3
    client.sendCTRL(ctrl)
    time.sleep(0.05)

# Utilities for getting common DREFs
def get_local_x(client):
    """
    Get the value of the aircraft's x-position in local coordinates
    """
    return client.getDREF("sim/flightmodel/position/local_x")[0]

def get_local_y(client):
    """
    Get the value of the aircraft's y-position in local coordinates
    """
    return client.getDREF("sim/flightmodel/position/local_y")[0]

def get_local_z(client):
    """
    Get the value of the aircraft's z-position in local coordinates
    """
    return client.getDREF("sim/flightmodel/position/local_z")[0]

def get_local_vx(client):
    """
    Get the value of the aircraft's x-velocity in local coordinates
    """
    return client.getDREF("sim/flightmodel/position/local_vx")[0]

def get_local_vy(client):
    """
    Get the value of the aircraft's y-velocity in local coordinates
    """
    return client.getDREF("sim/flightmodel/position/local_vy")[0]

def get_local_vz(client):
    """
    Get the value of the aircraft's z-velocity in local coordinates
    """
    return client.getDREF("sim/flightmodel/position/local_vz")[0]


def get_heading(client):
    """
    Get the value of the aircraft's heading in degrees from the Z axis
    """
    return client.getDREF("sim/flightmodel/position/psi")[0]

def get_home_heading(client):
    """
    Get the value of the aircraft's heading in degrees from the runway
    """
    true_heading = get_heading(client)
    return true_heading - HOME_HEADING

def get_ground_velocity(client):
    return client.getDREF("sim/flightmodel/position/groundspeed")

# Utilities for sending common DREFs
def send_local_x(client, x):
    """
    Set the value of the aircraft's x-position in local coordinates
    """
    return client.sendDREF("sim/flightmodel/position/local_x", x)

def send_local_y(client, y):
    """
    Set the value of the aircraft's y-position in local coordinates
    """
    return client.sendDREF("sim/flightmodel/position/local_y", y)

def send_local_z(client, z):
    """
    Set the value of the aircraft's z-position in local coordinates
    """
    return client.sendDREF("sim/flightmodel/position/local_z", z)

def send_local_vx(client, vx):
    """
    Set the value of the aircraft's x-velocity in local coordinates
    """
    return client.sendDREF("sim/flightmodel/position/local_vx", vx)

def send_local_vy(client, vy):
    """
    Set the value of the aircraft's y-velocity in local coordinates
    """
    return client.sendDREF("sim/flightmodel/position/local_vy", vy)

def send_local_vz(client, vz):
    """
    Set the value of the aircraft's z-velocity in local coordinates
    """
    return client.sendDREF("sim/flightmodel/position/local_vz", vz)

def send_heading(client, psi):
    """
    Set the value of the aircraft's heading in degrees from the Z axis
    """
    return client.sendDREF("sim/flightmodel/position/psi", psi)

def send_true_heading(client, psi):
    p = [-998, -998, -998, -998, -998, psi, -998]
    return client.sendPOSI(p)

def send_ground_velocity(client, v):
    """
    """
    psi = get_heading(client) * (math.pi/180)
    vx = v*math.sin(psi)
    vz = -v*math.cos(psi)

    return client.sendDREFs(["sim/flightmodel/position/local_vx", "sim/flightmodel/position/local_vz"], [vx, vz])

def send_throttle(client, a):
    return client.sendCTRL([0, 0, 0, a])


# Utilities for shifting current values to new ones
def shift_local_x(client, dx):
    """
    Shift the value of the aircraft's x-position in local coordinates
    """
    x = get_local_x(client)
    return client.sendDREF("sim/flightmodel/position/local_x", x + dx)

def shift_local_y(client, dy):
    """
    Shift the value of the aircraft's y-position in local coordinates
    """
    y = get_local_y(client)
    return client.sendDREF("sim/flightmodel/position/local_y", y + dy)

def shift_local_z(client, dz):
    """
    Shift the value of the aircraft's z-position in local coordinates
    """
    z = get_local_z(client)
    return client.sendDREF("sim/flightmodel/position/local_z", z + dz)

def shift_heading(client, dpsi):
    """
    Shift the value of the aircraft's heading in local coordinates
    """
    psi = get_heading(client)
    return client.send_heading(psi + dpsi)

def shift_true_heading(client, dpsi):
    psi = client.getPOSI()[5]
    return client.send_true_heading(psi + dpsi)

def shift_forward(client, d):
    x = get_local_x(client)
    z = get_local_z(client)
    psi = get_heading(client) * (math.pi/180)
    dx = d*math.sin(psi)
    dz = -d*math.cos(psi)

    return client.sendDREFs(["sim/flightmodel/position/local_x", "sim/flightmodel/position/local_z"], [x + dx, z + dz])

def get_glideslope_points():
    '''
    Returns the glideslope points defined on the approach plate: https://aeronav.faa.gov/d-tpp/2302/00961RRZ4.PDF

    Defined in the Home reference frame (e.g., aligned to the runway)
    '''
    HUSMI = np.array([0, 0, 1005.84])
    UBGUY = np.array([0, 3889.2, 853.44])
    return HUSMI, UBGUY

def dist_from_glideslope(client):
    '''
    Computes the distance from the glideslope defined here: https://aeronav.faa.gov/d-tpp/2302/00961RRZ4.PDF

    Uses the "home" coordinate frame for the runway and MSL elevation
    '''

    HUSMI, UBGUY = get_glideslope_points()

    x, y, _ = getHomeState(client)
    z = client.getDREF("sim/flightmodel/position/elevation")[0]
    pos = np.array([x, y, z])
    return np.linalg.norm(np.cross(pos - HUSMI, pos - UBGUY)) / np.linalg.norm(UBGUY - HUSMI)

def heading_angle_conversion(deg):
    '''
    Converts a world-frame angle to a heading and vice-versa
    The transformation is its own inverse
    Args:
        deg: the angle (heading) in degrees
    Returns:
        float: the heading (angle) in degrees
    '''

    return (90 - deg) % 360

def body_frame_velocity(client):
    cos = math.cos
    sin = math.sin

    psi = client.getDREF('sim/flightmodel/position/psi')[0]
    theta = client.getDREF('sim/flightmodel/position/theta')[0]
    phi = client.getDREF('sim/flightmodel/position/phi')[0]

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

    vx = client.getDREF('sim/flightmodel/position/local_vx')[0]
    vy = client.getDREF('sim/flightmodel/position/local_vy')[0]
    vz = client.getDREF('sim/flightmodel/position/local_vz')[0]
    # local frame is East-Up-South and we convert to North-East-Down
    vel_vec = np.array([-vz, vx, -vy]).T

    return np.matmul(R, vel_vec)

def get_autoland_runway_thresh():
    '''
    The runway threshold along the y-axis in the home frame
    '''
    return 12464

def get_autoland_statevec(client):
    '''
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
    '''

    vel = body_frame_velocity(client)

    P = client.getDREF('sim/flightmodel/position/P')[0]
    Q = client.getDREF('sim/flightmodel/position/Q')[0]
    R = client.getDREF('sim/flightmodel/position/R')[0]

    phi = client.getDREF('sim/flightmodel/position/phi')[0]
    theta = client.getDREF('sim/flightmodel/position/theta')[0]
    psi = get_home_heading(client)

    # runway distances (different frame than home)
    home_x, home_y, _ = getHomeState(client)
    x = get_autoland_runway_thresh() - home_y
    y = -home_x
    h = client.getDREF('sim/flightmodel/position/elevation')[0]

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


