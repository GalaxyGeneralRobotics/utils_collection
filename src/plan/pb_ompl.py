# https://github.com/lyfkyle/pybullet_ompl
import numpy as np
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
import pybullet as p
from src.plan import utils
import time
from itertools import product
import copy

INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 5.0

class PbOMPLRobot():
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.

    Note:
    This parent class by default assumes that all joints are acutated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.
    '''
    def __init__(self, id, cid) -> None:
        # Public attributes
        self.id = id
        self.cid = cid

        # prune fixed joints
        all_joint_num = p.getNumJoints(id, physicsClientId=self.cid)
        all_joint_idx = list(range(all_joint_num))
        joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.joint_names = [p.getJointInfo(id, j, physicsClientId=self.cid)[1].decode() for j in joint_idx]
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        # print(self.joint_idx)
        self.joint_bounds = []

        self.reset()

    def _is_not_fixed(self, joint_idx):
        joint_info = p.getJointInfo(self.id, joint_idx, physicsClientId=self.cid)
        return joint_info[2] != p.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = p.getJointInfo(self.id, joint_id, physicsClientId=self.cid)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds
            if low < high:
                self.joint_bounds.append([low, high])
        # print("Joint bounds: {}".format(self.joint_bounds))
        return self.joint_bounds

    def get_cur_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0, physicsClientId=self.cid)

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class PbOMPL():
    def __init__(self, robot, cid, obstacles = [], fix_joints = []) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.robot = robot
        self.cid = cid
        self.robot_id = robot.id
        self.obstacles = obstacles
        self.fix_joints = fix_joints

        self.space = PbStateSpace(robot.num_dim - len(fix_joints))

        bounds = ob.RealVectorBounds(robot.num_dim - len(fix_joints))
        joint_bounds = self.robot.get_joint_bounds()
        j = 0
        for i, bound in enumerate(joint_bounds):
            if robot.joint_names[i] in fix_joints:
                continue
            bounds.setLow(j, bound[0])
            bounds.setHigh(j, bound[1])
            j += 1
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        # self.set_planner("RRT") # RRT by default

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set

        # check self-collision
        self.robot.set_state(self.add_fix_joints(self.state_to_list(state)))
        for link1, link2 in self.check_link_pairs:
            if utils.pairwise_link_collision(self.cid, self.robot_id, link1, self.robot_id, link2):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(self.cid, body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True, allow_collision_links = []):
        self.check_link_pairs = utils.get_self_link_pairs(self.cid, robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in utils.get_moving_links(self.cid, robot.id, robot.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "BiTRRT":
            self.planner = og.BiTRRT(self.ss.getSpaceInformation())
            self.planner.setRange(0.0)
            self.planner.setTempChangeFactor(0.1)
            self.planner.setInitTemperature(100)
            self.planner.setFrontierThreshold(0.0)
            self.planner.setFrontierNodeRatio(0.1)
            self.planner.setCostThreshold(1e300)
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        pdef = ob.ProblemDefinition(self.ss.getSpaceInformation())
        pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(self.ss.getSpaceInformation()))
        self.planner.setProblemDefinition(pdef)
        self.ss.setPlanner(self.planner)
        self.path_simplifier = og.PathSimplifier(self.ss.getSpaceInformation())

    def plan_start_goal(self, start, goal, allowed_time = DEFAULT_PLANNING_TIME, interpolate_num=None):
        '''
        plan a path to gaol from the given robot start state
        '''
        # print("start_planning")
        # print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        start_no_fix, goal_no_fix = self.remove_fix_joints(start), self.remove_fix_joints(goal)
        for i in range(len(start_no_fix)):
            s[i] = start_no_fix[i]
            g[i] = goal_no_fix[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(allowed_time)
        res = False
        sol_path_list = []
        if solved:
            interpolate_num = interpolate_num if not interpolate_num is None else INTERPOLATE_NUM
            print("Found solution: interpolating into {} segments".format(interpolate_num))
            # print the path to screen
            self.ss.simplifySolution()
            sol_path_geometric = self.ss.getSolutionPath()
            self.path_simplifier.ropeShortcutPath(sol_path_geometric)  # Apply shortcutting
            self.path_simplifier.reduceVertices(sol_path_geometric)  # Further simplify the path
            self.path_simplifier.simplifyMax(sol_path_geometric)  # Further simplify the path
            self.path_simplifier.smoothBSpline(sol_path_geometric)  # Apply B-Spline smoothing

            sol_path_geometric.interpolate(interpolate_num)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            for sol_path in sol_path_list:
                if not self.is_state_valid(sol_path):
                    return False, None
            sol_path_list = [self.add_fix_joints(state) for state in sol_path_states]
            res = np.abs(np.array(goal) - np.array(sol_path_list[-1])).max() < 2.5e-3
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list

    def plan(self, goal, allowed_time = DEFAULT_PLANNING_TIME, interpolate_num=None, fix_joints_value=dict()):
        '''
        plan a path to gaol from current robot state
        '''
        self.fix_joints_value = fix_joints_value
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time, interpolate_num=interpolate_num)

    def execute(self, path, dynamics=False):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, q[i],force=5 * 240., physicsClientId=self.cid)
            else:
                self.robot.set_state(q)
            p.stepSimulation(physicsClientId=self.cid)
            time.sleep(0.02)



    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]

    def add_fix_joints(self, state):
        j = 0
        result = []
        for n in self.robot.joint_names:
            if n in self.fix_joints_value:
                result.append(self.fix_joints_value[n])
            else:
                result.append(state[j])
                j += 1
        return result

    def remove_fix_joints(self, state):
        result = []
        for i, n in enumerate(self.robot.joint_names):
            if n in self.fix_joints_value:
                continue
            result.append(state[i])
        return result