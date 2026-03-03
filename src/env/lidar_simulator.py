
import sys
sys.path.append("../")
sys.path.append(".")
import math
import numpy as np
from shapely.geometry import LineString, Point, LinearRing, Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import affine_transform

from env.vehicle import State,VehicleBox

ORIGIN = Point((0,0))

class LidarSimlator():
    def __init__(self, 
        lidar_range:float = 10.0,
        lidar_num:int = 120
    ) -> None:
        '''
        Args:
            lidar_range(float): the max distance that the obstacle can be dietected.
            lidar_num(int): the beam num of the lidar simulation.
        '''
        self.lidar_range = lidar_range
        self.lidar_num = lidar_num
        self.lidar_lines = []
        for a in range(lidar_num):
            self.lidar_lines.append(LineString(((0,0), (math.cos(a*math.pi/lidar_num*2)*lidar_range,\
                 math.sin(a*math.pi/lidar_num*2)*lidar_range))))
        self.vehicle_boundary = self.get_vehicle_boundary()

    def get_observation(self, ego_state:State, obstacles:list):
        '''
        Get the lidar observation from the vehicle's view.

        Args:
            ego_state: the state of ego car.
            obstacles: the list of obstacles in map

        Return:
            lidar_obs(np.array): the lidar data in sequence of angle, with the length of lidar_num.
        '''

        ego_pos = (ego_state.loc.x, ego_state.loc.y, ego_state.heading)
        rotated_obstacles = self._rotate_and_filter_obstacles(ego_pos, obstacles)
        lidar_obs = self._fast_calc_lidar_obs(rotated_obstacles)
        return np.array(lidar_obs - self.vehicle_boundary)
    
    def get_vehicle_boundary(self, ):
        lidar_base = []
        for l in self.lidar_lines:
            distance = l.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    def _rotate_and_filter_obstacles(self, ego_pos:tuple, obstacles:list):
        '''
        Rotate the obstacles around the vehicle and remove the obstalces which is out of lidar range.
        '''
        x, y, theta = ego_pos
        a = math.cos(theta)
        b = math.sin(theta)
        x_off = -x*a - y*b
        y_off = x*b - y*a
        affine_mat = [a, b, -b, a, x_off, y_off]

        rotated_obstacles = []
        for obs in obstacles:
            if hasattr(obs, 'shape'):
                obs = obs.shape
            rotated_obs = affine_transform(obs, affine_mat)
            if rotated_obs.distance(ORIGIN) < self.lidar_range:
                rotated_obstacles.append(rotated_obs)

        return rotated_obstacles

    @staticmethod
    def _extract_rings(geom):
        """Extract LinearRings representing obstacle boundaries.

        Supports:
        - LinearRing
        - Polygon (exterior + interior rings)
        - MultiPolygon (all component polygons)
        - GeometryCollection (flatten)
        """
        if geom is None:
            return []
        if isinstance(geom, LinearRing):
            return [geom]
        if isinstance(geom, Polygon):
            rings = [geom.exterior]
            rings.extend(list(geom.interiors))
            return rings
        if isinstance(geom, MultiPolygon):
            rings = []
            for g in geom.geoms:
                rings.extend(LidarSimlator._extract_rings(g))
            return rings
        if isinstance(geom, GeometryCollection):
            rings = []
            for g in geom.geoms:
                rings.extend(LidarSimlator._extract_rings(g))
            return rings
        # Best-effort: use .boundary if available
        try:
            return LidarSimlator._extract_rings(geom.boundary)
        except Exception:
            return []
    
    def _fast_calc_lidar_obs(self, obstacles:list):
        '''
        Obtain the lidar observation making use of numpy builtin matrix acceleration.

        Parameter:
            obstacles ( list(LinearRing) ): the obstacles around the vehicle which have been transformed to the ego referrence.

        Return:
            lidar_obs (np.ndarray): in shape (LIDAR_NUM,)
        '''

        # Line 1: the lidar ray, ax + by + c = 0
        theta = np.array([a*math.pi/self.lidar_num*2 for a in range(self.lidar_num)]) # (120,)
        a = np.sin(theta).reshape(-1,1) # (120, 1)
        b = -np.cos(theta).reshape(-1,1)
        c = 0

        # Convert obstacle boundaries to edges ((x1,y1), (x2,y2))
        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in obstacles:
            for ring in self._extract_rings(obst):
                ring_coords = np.array(ring.coords)  # (n+1, 2)
                if ring_coords.shape[0] < 2:
                    continue
                x1s.extend(list(ring_coords[:-1, 0]))
                x2s.extend(list(ring_coords[1:, 0]))
                y1s.extend(list(ring_coords[:-1, 1]))
                y2s.extend(list(ring_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return np.ones((self.lidar_num))*self.lidar_range
        
        x1s = np.array(x1s).reshape(1,-1)
        x2s = np.array(x2s).reshape(1,-1)
        y1s = np.array(y1s).reshape(1,-1)
        y2s = np.array(y2s).reshape(1,-1)

        # Line 2: the obstacle edge, (y1-y2)x + (x2-x1)y + x1y2 - x2y1 = 0
        A = y1s - y2s
        B = x2s - x1s
        C = x1s*y2s - x2s*y1s

        # Intersection
        # | a  b | | x |   | -c |
        # | A  B | | y | = | -C |
        # det = aB - bA
        # x = (-c B - (-C) b) / det = (C b) / det
        # y = (a (-C) - (-c) A) / det = (-a C) / det
        
        det = a @ B - b @ A # (120, n_edge)
        
        # Filter parallel lines
        det[np.abs(det) < 1e-6] = 1e-6

        x = (C * b) / det
        y = (-C * a) / det

        # Check if intersection is on the segment
        # min(x1, x2) <= x <= max(x1, x2)
        # min(y1, y2) <= y <= max(y1, y2)
        # And on the ray (x*cos + y*sin > 0)
        
        is_on_segment = (x >= np.minimum(x1s, x2s) - 1e-6) & \
                        (x <= np.maximum(x1s, x2s) + 1e-6) & \
                        (y >= np.minimum(y1s, y2s) - 1e-6) & \
                        (y <= np.maximum(y1s, y2s) + 1e-6)
        
        is_on_ray = (x * np.cos(theta).reshape(-1,1) + y * np.sin(theta).reshape(-1,1)) > 0

        valid_intersection = is_on_segment & is_on_ray
        
        dist = np.sqrt(x**2 + y**2)
        dist[~valid_intersection] = self.lidar_range + 1.0
        
        min_dist = np.min(dist, axis=1)
        min_dist = np.clip(min_dist, 0, self.lidar_range)
        
        return min_dist
