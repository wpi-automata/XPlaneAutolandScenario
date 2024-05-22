import copy
import numpy as np

from xplane_autoland.xplane_connect.xpc3 import XPlaneConnect
from xplane_autoland.xplane_connect import xpc3_helper as xp


if __name__ == "__main__":
    # Run this with XPlane 11 running
    client = XPlaneConnect()
    xp.reset(client)

    def send_local_xz(client, x, z):
        xp.send_local_x(client, x)
        xp.send_local_z(client, z)

    # local coordinate frame has y as up
    def get_local_xz(client):
        return [xp.get_local_x(client), xp.get_local_z(client)]

    start_pos_local = get_local_xz(client)
    start_pos_autoland = xp.get_autoland_statevec(client)[-3:-1]

    l0 = np.array(start_pos_local).reshape(2, 1)
    v0 = np.array(start_pos_autoland).reshape(2, 1)

    send_local_xz(client, l0[0], l0[1] + 1)
    xl, zl = get_local_xz(client)
    l1 = np.array([[xl], [zl]])
    state = xp.get_autoland_statevec(client)
    v1 = state[-3:-1].reshape((2, 1))

    l = l1 - l0
    v = v1 - v0

    theta = np.arccos(
        np.dot(l.flatten(), v.flatten()) / (np.linalg.norm(l) * np.linalg.norm(v))
    )
    print(f"Got theta = {theta}")
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = l0 - R @ v0
    t = t.reshape((2, 1))
    print(f"Translation = {t}")
