import numpy as np
from RRTTree import RRTTree
import time

class RRTPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode #edge extension mode for RRT and RRTStar
        self.goal_prob = goal_prob #probability to draw goal vertex for RRT and RRTStar

    def plan(self):
        '''
        Compute and return the plan. 
        The function should return a numpy array 
        containing the states (positions) of the robot.
        '''
        start_time = time.time()
        
        # initialize an empty plan.
        plan = []
        
        # TODO: Task 4.4
        #initializing the tree with the start position
        start_state = self.planning_env.start
        goal_state = self.planning_env.goal

        self.tree.add_vertex(self.planning_env.start)
        start_id = self.tree.get_idx_for_state(start_state)

        #times = []
        #qualities = []
        #total_run_time = 100
        #last_time = start_time

        while not self.tree.is_goal_exists(goal_state):
        #while time.time() - start_time < total_run_time:
            state_rand = sample_random(self.goal_prob, self.planning_env, self.planning_env.goal)
            _, state_near = self.tree.get_nearest_state(state_rand)
            state_new = self.extend(state_near, state_rand)
            if self.planning_env.edge_validity_checker(state_near, state_new):
                v_near_id = self.tree.get_idx_for_state(state_near)
                v_new_id = self.tree.add_vertex(state_new)
                
                dist = self.planning_env.compute_distance(state_near, state_new)
                self.tree.add_edge(v_near_id, v_new_id, dist)
            
            """goal_id = self.tree.get_idx_for_state(goal_state)
            if goal_id != None:
                if time.time() - last_time > 10:        
                    last_time = time.time()
                    curr_id = goal_id
                    while (curr_id != start_id):
                        plan.append(self.tree.vertices[curr_id].state)
                        curr_id = self.tree.edges[curr_id]
                    plan.append(start_state)
                    times.append(time.time()-start_time)
                    qualities.append(self.compute_cost(plan))                    
                    plan = []"""
        
        end_time = time.time()-start_time
        #times.append(end_time)

        #constructing the plan from the goal to the start
        curr_id = self.tree.get_idx_for_state(goal_state)
        start_id = self.tree.get_idx_for_state(start_state)#should be 0 always
        if curr_id != None:
            while (curr_id != start_id):
                plan.append(self.tree.vertices[curr_id].state)
                curr_id = self.tree.edges[curr_id]
            plan.append(start_state)

        total_cost = self.compute_cost(plan)        
        #qualities.append(total_cost)
                
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(total_cost))
        print('Total time: {:.2f}'.format(end_time))

        #print('qualities: ', qualities)
        #print('times: ', times)

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 4.4
        cost = 0.0
        for i in range(len(plan) - 1):
            cost += self.planning_env.compute_distance(plan[i], plan[i + 1])
        return cost

    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''
        # TODO: Task 4.4

        if self.ext_mode == "E2":
            step_size = 10
            dist = self.planning_env.compute_distance(near_state, rand_state)
            if step_size > dist:
                return rand_state            
            intervals = int(np.ceil(dist / step_size))
            new_states = np.linspace(near_state, rand_state, intervals)
            rand_state = new_states[1]

        return rand_state

#our function
def sample_random(goal_prob, env, goal) -> np.array:
    if np.random.rand() < goal_prob:
        return goal
    else:# With probability 1 - p_bias, sample randomly within joint limits
        x = np.random.uniform(env.xlimit[0], env.xlimit[1])
        y = np.random.uniform(env.ylimit[0], env.ylimit[1])
        while not env.state_validity_checker((x, y)):
            x = np.random.uniform(env.xlimit[0], env.xlimit[1])
            y = np.random.uniform(env.ylimit[0], env.ylimit[1])   
        return np.array((x, y))