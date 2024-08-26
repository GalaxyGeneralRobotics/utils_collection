# https://github.com/lyfkyle/pybullet_ompl
import os
import sys

ROOT_PATH = os.path.abspath(__file__)
for _ in range(3):
    ROOT_PATH = os.path.dirname(ROOT_PATH)
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)

import pybullet as p
import math
import sys
import pybullet_data
from transforms3d.quaternions import quat2mat, mat2quat
from src.plan import pb_ompl
from src.utils.vis_plotly import Vis
from src.utils.config import DotDict
from src.utils.utils import to_list
import numpy as np

class Planner:
    def __init__(self, config, vis: bool=0, fix_joints=[]):
        self.obstacles = []

        self.cid = p.connect(p.GUI if vis else p.DIRECT)
        if vis:
            p.resetDebugVisualizerCamera(2, 90, -45, [0, 0, 0], physicsClientId=self.cid)
        p.setGravity(0, 0, 0, physicsClientId=self.cid)
        p.setTimeStep(1./24000000., physicsClientId=self.cid)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        # p.loadURDF("plane.urdf", physicsClientId=self.cid)

        # load robot
        robot_id = p.loadURDF(config.urdf, (0,0,0), useFixedBase = 1, physicsClientId=self.cid)
        robot = pb_ompl.PbOMPLRobot(robot_id, self.cid)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.cid, self.obstacles, fix_joints=fix_joints)
        # self.pb_ompl_interface.set_planner("BITStar")
        self.pb_ompl_interface.set_planner("BiTRRT")

        # add obstacles
        self.add_obstacles(config.obj)
        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle, physicsClientId=self.cid)
        self.obstacles = []

    def add_obstacles(self, obj_config):
        # add box
        for obj in obj_config:
            if obj.type == 'mesh':
                obj_id = p.loadURDF(obj.urdf, basePosition=to_list(obj.pos), baseOrientation=to_list(obj.quat[[1,2,3,0]]), physicsClientId=self.cid)
                self.obstacles.append(obj_id)
            elif obj.type == 'box':
                colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=to_list(obj.size / 2), physicsClientId=self.cid)
                box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=to_list(obj.pos), baseOrientation=to_list(obj.quat[[1,2,3,0]]), physicsClientId=self.cid)
                self.obstacles.append(box_id)

    def plan(self, start=None, goal=None, exec=False, interpolate_num=None, fix_joints_value=dict()):
        if start is None:
            start = [0,0,0,-1,0,1.5,0, 0.02, 0.02]
        if goal is None:
            goal = [1,0,0,-1,0,1.5,0, 0.02, 0.02]
        # goal = [0,1.5,0,-0.1,0,0.2,0, 0.02, 0.02]

        self.pb_ompl_interface.fix_joints_value = fix_joints_value
        start, goal = to_list(start), to_list(goal)
        for name, pose in [('start', start), ('goal', goal)]:
            if not self.pb_ompl_interface.is_state_valid(pose):
                print(f'unreachable {name}')
                return False, None

        self.robot.set_state(start)
        res, path = self.pb_ompl_interface.plan(goal, interpolate_num=interpolate_num, fix_joints_value=fix_joints_value)
        if res:
            path = np.array(path)
            if exec:
                self.pb_ompl_interface.execute(path)
        return res, path

    def close(self):
        p.disconnect(physicsClientId=self.cid)

if __name__ == '__main__':
    cfg = DotDict(
        urdf='robot_models/franka/franka_with_gripper_extensions.urdf',
        obj=[]
    )
    env = Planner(cfg, vis=0)
    env.plan([0,0,0,-1,0,1.5,0, 0.02, 0.02], [1,0,0,-1,0,1.5,0, 0.02, 0.02])
    env = Planner(cfg, vis=1)
    env.plan(exec=True)