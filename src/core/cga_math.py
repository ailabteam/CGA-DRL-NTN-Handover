import numpy as np
import math
from clifford.g3c import *
from clifford.tools.g3c import *

R_EARTH = 6371.0 

class CGAEngine:
    def __init__(self):
        self.e_inf = einf
        self.e_o = eo
        self.earth_sphere = self.e_o - 0.5 * (R_EARTH**2) * self.e_inf

    def latlon_to_cga(self, lat, lon, alt=0.0):
        r = R_EARTH + alt
        phi = np.radians(lat)
        theta = np.radians(lon)
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        x_vec = x*e1 + y*e2 + z*e3
        return up(x_vec)

    def get_distance_sq(self, P1, P2):
        inner_prod = P1 | P2 
        val = inner_prod[0]
        dist_sq = -2.0 * val
        return abs(dist_sq) 

    def propagate_satellite(self, sat_point, angular_velocity_rad, dt, orbit_normal_bivector):
        angle = angular_velocity_rad * dt
        rotor = math.e**(-0.5 * angle * orbit_normal_bivector)
        new_pos = rotor * sat_point * ~rotor
        return new_pos

    def check_visibility_fast(self, user_point, sat_point, min_elevation_deg=10.0):
        u_vec_ga = user_point(1) 
        s_vec_ga = sat_point(1)
        u_arr = np.array([u_vec_ga[e1], u_vec_ga[e2], u_vec_ga[e3]])
        s_arr = np.array([s_vec_ga[e1], s_vec_ga[e2], s_vec_ga[e3]])
        r_u = np.linalg.norm(u_arr)
        r_s = np.linalg.norm(s_arr)
        if r_u == 0 or r_s == 0: return False
        cos_gamma = np.dot(u_arr, s_arr) / (r_u * r_s)
        gamma_rad = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        horizon_gamma = np.arccos(R_EARTH / r_s)
        min_el_rad = np.radians(min_elevation_deg)
        if gamma_rad < (horizon_gamma - min_el_rad/4.0): 
             return True
        return False

    def to_features(self, user_point, sat_point):
        d_sq = self.get_distance_sq(user_point, sat_point)
        dist = np.sqrt(d_sq)
        feat_dist = dist / 2000.0
        
        u_vec_ga = user_point(1) 
        s_vec_ga = sat_point(1)
        u_arr = np.array([u_vec_ga[e1], u_vec_ga[e2], u_vec_ga[e3]])
        s_arr = np.array([s_vec_ga[e1], s_vec_ga[e2], s_vec_ga[e3]])
        norm_prod = (np.linalg.norm(u_arr) * np.linalg.norm(s_arr))
        if norm_prod == 0: cos_gamma = 0.0
        else: cos_gamma = np.dot(u_arr, s_arr) / norm_prod
        
        return np.array([feat_dist, cos_gamma], dtype=np.float32)

    def get_radial_velocity(self, user_point, sat_point, sat_velocity_vector):
        u_vec = user_point(1)
        s_vec = sat_point(1)
        r_vec = s_vec - u_vec 
        r_sq = (r_vec | r_vec)[0]
        r_norm = math.sqrt(float(r_sq))
        if r_norm == 0: return 0.0
        dot_prod = float((sat_velocity_vector | r_vec)[0])
        v_rad = dot_prod / r_norm
        return v_rad
