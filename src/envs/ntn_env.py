# src/envs/ntn_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.core.cga_math import CGAEngine
from src.core.constellation import WalkerConstellation
from clifford.g3c import e1, e2, e3 # Import để truy cập coord cho baseline

class SatelliteHandoverEnv(gym.Env):
    def __init__(self, k_nearest=5, max_steps=1000, feature_type='cga'):
        """
        feature_type: 
            - 'cga': Dùng đặc trưng bất biến (Distance, Cosine).
            - 'xyz': Dùng toạ độ Euclide tương đối (dx, dy, dz). -> BASELINE
        """
        super(SatelliteHandoverEnv, self).__init__()
        
        self.k_nearest = k_nearest
        self.max_steps = max_steps
        self.feature_type = feature_type # 'cga' or 'xyz'
        self.step_count = 0
        
        self.cga = CGAEngine()
        self.constellation = WalkerConstellation(self.cga, total_sats=66, n_planes=6, inclination=86.4, altitude=780.0)
        self.user_pos = self.cga.latlon_to_cga(21.028, 105.854, 0.0)
        
        self.action_space = spaces.Discrete(k_nearest)
        
        # Observation Space Config
        # CGA: [dist, cos, conn] -> 3 features
        # XYZ: [dx, dy, dz, conn] -> 4 features
        if self.feature_type == 'cga':
            self.feat_dim = 3
        else:
            self.feat_dim = 4
            
        self.obs_dim = k_nearest * self.feat_dim
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)
        
        self.current_sat_id = -1 
        self.last_sat_id = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.constellation = WalkerConstellation(self.cga)
        self.step_count = 0
        self.current_sat_id = -1
        self.last_sat_id = -1
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        dt = 10.0
        self.constellation.propagate(dt)
        
        candidates = self._get_candidates()
        
        reward = 0
        done = False
        info = {}
        
        if len(candidates) == 0:
            reward = -10.0
            self.current_sat_id = -1
        else:
            selected_idx = min(action, len(candidates)-1)
            target_sat = candidates[selected_idx]
            target_id = target_sat['id']
            
            # Reward function giữ nguyên cho cả 2 trường hợp để so sánh công bằng
            # Vẫn dùng cosine (chất lượng tín hiệu) làm thước đo
            quality_reward = target_sat['features'][1] * 2.0 
            
            ho_penalty = 0.0
            if self.current_sat_id != -1 and target_id != self.current_sat_id:
                ho_penalty = -0.5 
                
            reward = quality_reward + ho_penalty
            
            self.last_sat_id = self.current_sat_id
            self.current_sat_id = target_id
            
        if self.step_count >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, info

    def _get_candidates(self):
        candidates = []
        for sat in self.constellation.satellites:
            if self.cga.check_visibility_fast(self.user_pos, sat['pos']):
                # Luôn tính CGA features để dùng cho việc sort và reward
                feats_cga = self.cga.to_features(self.user_pos, sat['pos'])
                
                # Tính XYZ features cho Baseline
                # Lấy vector tương đối: Sat - User
                u_vec = self.user_pos(1)
                s_vec = sat['pos'][1]
                dx = float(s_vec[e1] - u_vec[e1]) / 6371.0 # Normalize theo R_Earth
                dy = float(s_vec[e2] - u_vec[e2]) / 6371.0
                dz = float(s_vec[e3] - u_vec[e3]) / 6371.0
                
                feats_xyz = np.array([dx, dy, dz], dtype=np.float32)

                candidates.append({
                    'id': sat['id'],
                    'pos': sat['pos'],
                    'features': feats_cga, # Vẫn giữ để tính reward
                    'features_xyz': feats_xyz
                })
        
        candidates.sort(key=lambda x: x['features'][0])
        return candidates

    def _get_obs(self):
        candidates = self._get_candidates()
        obs_vec = np.zeros(self.obs_dim, dtype=np.float32)
        
        for i in range(self.k_nearest):
            if i < len(candidates):
                cand = candidates[i]
                conn_flag = 1.0 if cand['id'] == self.current_sat_id else 0.0
                
                if self.feature_type == 'cga':
                    # [dist, cos, conn]
                    obs_vec[i*3] = cand['features'][0] 
                    obs_vec[i*3+1] = cand['features'][1]
                    obs_vec[i*3+2] = conn_flag
                else: # xyz baseline
                    # [dx, dy, dz, conn]
                    obs_vec[i*4] = cand['features_xyz'][0]
                    obs_vec[i*4+1] = cand['features_xyz'][1]
                    obs_vec[i*4+2] = cand['features_xyz'][2]
                    obs_vec[i*4+3] = conn_flag
            else:
                # Padding cho trường hợp thiếu vệ tinh
                if self.feature_type == 'cga':
                    obs_vec[i*3] = 2.0; obs_vec[i*3+1] = -1.0; obs_vec[i*3+2] = 0.0
                else:
                    obs_vec[i*4] = 0.0; obs_vec[i*4+1] = 0.0; obs_vec[i*4+2] = 0.0; obs_vec[i*4+3] = 0.0
                
        return obs_vec
