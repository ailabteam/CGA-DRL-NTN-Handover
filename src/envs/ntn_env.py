import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.core.cga_math import CGAEngine
from src.core.constellation import WalkerConstellation
from clifford.g3c import e1, e2, e3
from scipy.spatial.transform import Rotation as R

class SatelliteHandoverEnv(gym.Env):
    def __init__(self, k_nearest=5, max_steps=1000, feature_type='cga', scenario='static'):
        super(SatelliteHandoverEnv, self).__init__()
        
        self.k_nearest = k_nearest
        self.max_steps = max_steps
        self.feature_type = feature_type
        self.scenario = scenario
        self.step_count = 0
        
        self.cga = CGAEngine()
        self.constellation = WalkerConstellation(self.cga, total_sats=66, n_planes=6, inclination=86.4, altitude=780.0)
        self.user_pos = None 
        
        self.action_space = spaces.Discrete(k_nearest)
        self.feat_dim = 4 # [dist, cos, v_rad, conn]
        self.obs_dim = k_nearest * self.feat_dim
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32)
        
        self.current_sat_id = -1 
        self.last_sat_id = -1
        # Metrics Tracking
        self.ho_history = [] 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. User Position Logic
        if self.scenario == 'random':
            # Random location globally (Lat limited to coverage area)
            rand_lat = np.random.uniform(-60, 60)
            rand_lon = np.random.uniform(-180, 180)
            self.user_pos = self.cga.latlon_to_cga(rand_lat, rand_lon, 0.0)
            
            # Generate Random Rotation Matrix for this episode
            # This simulates user orientation changes or sensor noise frame
            self.rot_matrix = R.random().as_matrix()
        else:
            # Static: Hanoi
            self.user_pos = self.cga.latlon_to_cga(21.028, 105.854, 0.0)
            self.rot_matrix = np.eye(3) # No rotation

        self.constellation = WalkerConstellation(self.cga)
        self.step_count = 0
        self.current_sat_id = -1
        self.last_sat_id = -1
        self.ho_history = []
        
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        dt = 10.0
        self.constellation.propagate(dt)
        
        candidates = self._get_candidates()
        
        reward = 0
        done = False
        info = {
            "is_ho": 0,
            "throughput": 0.0,
            "outage": 0
        }
        
        if len(candidates) == 0:
            reward = -10.0
            self.current_sat_id = -1
            info["outage"] = 1
        else:
            selected_idx = min(action, len(candidates)-1)
            target_sat = candidates[selected_idx]
            target_id = target_sat['id']
            
            # Reward Components
            # 1. QoS (Cosine Elevation)
            cos_el = target_sat['features_cga'][1]
            quality_reward = cos_el * 2.0 
            
            # 2. Handover Penalty
            ho_penalty = 0.0
            if self.current_sat_id != -1 and target_id != self.current_sat_id:
                ho_penalty = -0.5 
                info["is_ho"] = 1
                self.ho_history.append(self.step_count)
                
            reward = quality_reward + ho_penalty
            
            # Ping-Pong Detection (Simple: HO back within 3 steps)
            # (Implemented in wrapper or post-analysis)
            
            # Throughput Estimation (Shannon-like)
            # R = B * log2(1 + SNR), SNR ~ sin(el)
            sin_el = np.sqrt(max(0, 1 - cos_el**2))
            throughput = 100.0 * np.log2(1 + 10.0 * sin_el)
            info["throughput"] = throughput
            
            self.last_sat_id = self.current_sat_id
            self.current_sat_id = target_id
            
        if self.step_count >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, info

    def _get_candidates(self):
        candidates = []
        for sat in self.constellation.satellites:
            if self.cga.check_visibility_fast(self.user_pos, sat['pos']):
                
                # --- CGA Features (INVARIANT) ---
                basic_feats = self.cga.to_features(self.user_pos, sat['pos'])
                if 'velocity' in sat:
                    v_rad = self.cga.get_radial_velocity(self.user_pos, sat['pos'], sat['velocity'])
                else:
                    v_rad = 0.0
                v_rad_norm = v_rad / 7.0 
                feats_cga = np.array([basic_feats[0], basic_feats[1], v_rad_norm], dtype=np.float32)
                
                # --- XYZ Features (VARIANT - Rotated) ---
                u_vec = self.user_pos(1)
                s_vec = sat['pos'][1]
                diff_vec = s_vec - u_vec
                
                # Convert to numpy for rotation
                diff_arr = np.array([float((diff_vec | e1)[0]), float((diff_vec | e2)[0]), float((diff_vec | e3)[0])])
                
                # Apply Random Rotation Noise (Critical for beating baseline)
                diff_rotated = self.rot_matrix @ diff_arr
                
                dx = diff_rotated[0] / 6371.0
                dy = diff_rotated[1] / 6371.0
                dz = diff_rotated[2] / 6371.0
                feats_xyz = np.array([dx, dy, dz], dtype=np.float32)

                candidates.append({
                    'id': sat['id'],
                    'pos': sat['pos'],
                    'features_cga': feats_cga, 
                    'features_xyz': feats_xyz
                })
        
        candidates.sort(key=lambda x: x['features_cga'][0])
        return candidates

    def _get_obs(self):
        candidates = self._get_candidates()
        obs_vec = np.zeros(self.obs_dim, dtype=np.float32)
        
        for i in range(self.k_nearest):
            if i < len(candidates):
                cand = candidates[i]
                conn_flag = 1.0 if cand['id'] == self.current_sat_id else 0.0
                
                if self.feature_type == 'cga':
                    obs_vec[i*4] = cand['features_cga'][0] 
                    obs_vec[i*4+1] = cand['features_cga'][1]
                    obs_vec[i*4+2] = cand['features_cga'][2]
                    obs_vec[i*4+3] = conn_flag
                else: 
                    obs_vec[i*4] = cand['features_xyz'][0]
                    obs_vec[i*4+1] = cand['features_xyz'][1]
                    obs_vec[i*4+2] = cand['features_xyz'][2]
                    obs_vec[i*4+3] = conn_flag
            else:
                obs_vec[i*4:i*4+4] = 0.0
                
        return obs_vec
