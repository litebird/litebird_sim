# -*- encoding: utf-8 -*-
import numpy as np

from astropy.coordinates import BarycentricMeanEcliptic
from astropy.coordinates import SkyCoord

"""The coordinate system used by the framework"""
DEFAULT_COORDINATE_SYSTEM = BarycentricMeanEcliptic()

"""The time scale used by the framework"""
DEFAULT_TIME_SCALE = "tdb"

e2g = (
    SkyCoord(
        x=[1., 0., 0.],
        y=[0., 1., 0.],
        z=[0., 0., 1.],
        frame=DEFAULT_COORDINATE_SYSTEM.name,
        representation_type="cartesian",
    )
    .transform_to("galactic")
    .data.to_cartesian()
    .get_xyz()
    .value
)

def ang2vec(theta, phi):
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    vec = np.empty((3, ct.size), np.float64)
    vec[0, :] = st * cp
    vec[1, :] = st * sp
    vec[2, :] = ct
    return vec.squeeze()

def vec2ang(vx, vy, vz):
    r = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    ang = np.empty((2, r.size))
    ang[0, :] = np.arccos(vz / r)
    ang[1, :] = np.arctan2(vy, vx)
    return ang.squeeze()

def rotate_coordinates_e2g(pointings):
    vec = np.tensordot(e2g, ang2vec(pointings[:,0], pointings[:,1]), axes=(1, 0))
    north_pole = np.tensordot(e2g, [0.0, 0.0, 1.0], axes=(1, 0))
    sinalpha = north_pole[0] * vec[1] - north_pole[1] * vec[0]
    cosalpha = north_pole[2] - vec[2] * np.dot(north_pole, vec)
    pointings[:,0], pointings[:,1] = vec2ang(vec[0], vec[1], vec[2])
    pointings[:,2] += np.arctan2(sinalpha, cosalpha)
    return

