# from https://colab.research.google.com/github/google-deepmind/dm_control/blob/main/tutorial.ipynb

import os
import sys

ROOT_PATH = os.path.abspath(__file__)
for _ in range(3):
    ROOT_PATH = os.path.dirname(ROOT_PATH)
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from PIL import Image
import torch
from transforms3d.quaternions import quat2mat

from dm_control import mujoco
from dm_control import mjcf

from src.utils.config import to_dot_dict
from src.mujoco.env_utils import file_to_str
from src.utils.utils import depth_img_to_xyz, to_numpy
from src.utils.vis_plotly import Vis

class Env:
    def __init__(self, config):
        self.config = config
        self.root = mjcf.RootElement()
        self.root.option.set_attributes(impratio=10, integrator="implicitfast", cone="elliptic", noslip_iterations=2)
        self.root.default.geom.set_attributes(solref=[0.001, 1])
        # self.root.option.set_attributes(timestep=0.002, integrator="implicitfast", cone="pyramidal", noslip_iterations=3)
        self.world = self.root.worldbody
        self.meshes = set()

        for obj in config.obj:
            self.add_obj(obj)
        if 'robot' in config:
            self.add_robot(config.robot)
        if 'camera' in config:
            self.add_camera(config.camera)
        self.root.keyframe.key.clear()
        self.physics = mjcf.Physics.from_mjcf_model(self.root)
        self.objs = {obj['name']: self.physics.data.body(obj['name']) for obj in config.obj}
        if config.only_obj_gravity:
            self.physics.model.opt.gravity = (0, 0, 0)
        self.actuators = [self.physics.data.actuator(a) for a in self.config.actuators]
        self.joints = [self.physics.data.joint(a) for a in self.config.joints]

        self.substeps = round(config.dt / self.physics.timestep())
        assert config.dt == self.substeps * self.physics.timestep()

    def add_obj(self, obj_config):
        if obj_config.type == 'mesh':
            obj_body = self.world.add('body', name=obj_config.name, pos=obj_config.pos, quat=obj_config.quat)
            if not obj_config.fixed:
                obj_body.add('joint', type='free', name=f'object_joint_{obj_config.name}', stiffness=0, damping=0, frictionloss=0, armature=0.0)
                # obj_body.add('freejoint')
            for p in obj_config.path:
                obj_body.add('geom', type='mesh', mesh=file_to_str(p), condim=4, mass=self.config.obj_mass/len(obj_config.path), conaffinity=1, contype=1, friction=[0.6, 0.3, 1.0])
                if not p in self.meshes:
                    self.root.asset.add('mesh', name=file_to_str(p), file=p)
                    self.meshes.add(p)
        elif obj_config.type == 'box':
            obj_body = self.world.add('body', name=obj_config.name, pos=obj_config.pos, quat=obj_config.quat)
            if not obj_config.fixed:
                obj_body.add('freejoint')
            obj_body.add('geom', type='box', size=obj_config.size / 2, condim=4, friction=[0.6, 0.3, 1.0])
    
    def add_robot(self, robot_config):
        mjcf_panda = mjcf.from_path(robot_config)
        robot = self.world.attach(mjcf_panda)
        robot.pos = np.array([0., 0., 0.])
        robot.quat = np.array([1., 0., 0., 0.])
    
    def add_camera(self, camera_config):
        self.world.add('camera', name='camera', pos=camera_config.pos, quat=camera_config.quat, fovy=camera_config.fovy)
        self.root.visual.map.znear = 1e-5
    
    def reset(self, init_pose, require=dict()):
        init_pose = to_numpy(init_pose)
        self.physics.reset()
        if self.config.only_obj_gravity:
            pseudo_gravity = np.zeros(6)
            pseudo_gravity[2] = -9.81 * self.config.obj_mass
            for obj in self.objs.values():
                obj.xfrc_applied = pseudo_gravity
        for i in range(len(self.joints)):
            self.joints[i].qpos[:] = init_pose[i]
            self.joints[i].qvel[:] = 0
            self.joints[i].qacc[:] = 0
            self.actuators[i].ctrl[:] = init_pose[i]
        return self.get_state(require)
    
    def get_state(self, require=dict()):
        qpos = np.concatenate([j.qpos[:] for j in self.joints])
        obj_pose = {}
        for name, obj in self.objs.items():
            pose = np.eye(4)
            pose[:3, 3] = obj.xpos
            pose[:3, :3] = obj.xmat.reshape(3, 3)
            obj_pose[name] = pose
        state = dict(
            qpos=qpos,
            obj_pose=obj_pose,
        ) 
        scene_option = mujoco.wrapper.core.MjvOption()
        # scene_option.geomgroup[3] = 1
        # scene_option.geomgroup[2] = 0
        if require.get('rgb', True):
            pixels = self.physics.render(scene_option=scene_option, camera_id='camera', height=self.config.camera.h, width=self.config.camera.w)
            state['rgb'] = pixels
        if require.get('depth', True) or require.get('pc', True):
            depth = self.physics.render(scene_option=scene_option, camera_id='camera', height=self.config.camera.h, width=self.config.camera.w, depth=True)
            state['depth'] = depth
            xyz = depth_img_to_xyz(depth, self.config.camera.intrinsics).reshape(-1, 3)
            xyz[..., 1:] *= -1
            pc = np.einsum('ab,nb->na', quat2mat(self.config.camera.quat), xyz) + self.config.camera.pos
            state['pc'] = pc
        return state


    def act(self, action, set_act=False, require=dict()):
        action = to_numpy(action)
        for i in range(len(self.joints)):
            if set_act:
                self.joints[i].qpos[:] = action[i]
            else:
                self.actuators[i].ctrl[:] = action[i]
        for _ in range(self.substeps):
            self.physics.step()
        state = self.get_state(require)
        return state

class BatchEnv:
    def __init__(self, configs):
        self.n = len(configs)
        self.configs = configs
        self.envs = [Env(config) for config in configs]

    def reset(self, init_poses, require=dict()):
        return [e.reset(init_pose, require) for e, init_pose in zip(self.envs, init_poses)]

    def act(self, actions, set_act=False, require=dict()):
        return [e.act(action, set_act, require) for e, action in zip(self.envs, actions)]