import numpy as np
import heapdict


class AStarPlanner(object):
    def __init__(self, planning_env, h_weight=20.0):
        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.planning_env = planning_env
        self.expanded_nodes = []
        self.parent = []
        self.h_weight = h_weight
        self.open = heapdict.heapdict()
        self.close = heapdict.heapdict()
        self.update_node_open=0
        self.update_node_close = 0

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        # initial_state= self.state_neighbors(self.planning_env.start)
        initial_state = self.planning_env.start
        Initial_node = A_star_Node(initial_state)
        Initial_node.h_value = self.planning_env.compute_heuristic(Initial_node.state)
        Initial_node.f_value = self.f_func(Initial_node.h_value, 0, self.h_weight)
        Initial_node.parent = None

        self.open[(tuple(initial_state))] = (Initial_node.f_value, Initial_node.parent, Initial_node.cost)

        if np.array_equal(self.planning_env.goal, initial_state):
            return [], 0, 0
        i = 0
        while self.open:
            i += 1
            #if (i % 1000) == 0:
            #    print(i)

            state, node_heap = self.open.popitem()
            node = node_heap

            self.close[state] = (node[0], str(node[1]))
            if state not in self.expanded_nodes: # Add to combined close heapdict
                self.expanded_nodes.append(state)

            if np.array_equal(self.planning_env.goal, np.array(state)):
                # if np.array_equal(self.planning_env.goal, node.state):
                Plan = self.path(state)
                #print("Num of iteration: ",i)
                #print("Num of expanded nedes: ", len(self.expanded_nodes))
                Total_cost = node[2]
                #print("Total_cost: ", Total_cost)
                #print("self.update_node_open= ",self.update_node_open)
                #print("self.update_node_close = ", self.update_node_close)
                return Plan, Total_cost
            node_neighbors = self.state_neighbors(np.array(state))
            for neighbor in node_neighbors:
                new_node = A_star_Node(neighbor)
                new_node.parent = state
                new_node.cost = self.planning_env.compute_distance(new_node.state, state) + node[2]
                new_node.h_value = self.planning_env.compute_heuristic(new_node.state)
                new_node.f_value = self.f_func(new_node.h_value, new_node.cost, self.h_weight)
                if ((tuple(new_node.state))) not in self.close:  # Check in close heapdict
                    if (tuple(new_node.state)) not in self.open:

                        self.open[(tuple(new_node.state))] = (new_node.f_value, new_node.parent, new_node.cost)
                    else:
                        existing_priority = self.open[(tuple(new_node.state))]
                        if new_node.f_value < existing_priority[0]:
                            del self.open[(tuple(new_node.state))]
                            self.open[(tuple(new_node.state))] = (new_node.f_value, new_node.parent, new_node.cost)
                            self.update_node_open+=1

                else:

                    f_value = self.close[(tuple(new_node.state))]

                    if new_node.f_value < f_value[0]:

                        del self.close[(tuple(new_node.state))]
                        self.open[(tuple(new_node.state))] = (new_node.f_value, new_node.parent, new_node.cost)
                        self.update_node_close += 1


    # TODO: Task 4.3
    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.

        '''
        # used for visualizing the expanded nodes
        return self.expanded_nodes

    def state_neighbors(self, current_state):
        # Define 8-connected neighborhood offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1), (0, 1),
                   (1, -1), (1, 0), (1, 1)]

        neighbors = []

        # Generate neighboring states
        for dx, dy in offsets:
            neighbor_state = np.array([current_state[0] + dx, current_state[1] + dy])

            # Check if the neighbor state is valid and the edge is collision-free
            if self.planning_env.state_validity_checker(
                    state=neighbor_state) and self.planning_env.edge_validity_checker(current_state, neighbor_state):
                neighbors.append(neighbor_state)

        return np.array(neighbors)

    def f_func(self, h_value, g_value, h_weight):  # f value calculator
        f_score = (h_weight * h_value) + (g_value)
        return f_score

    def path(self, node):
        path = []
        path.append(node)
        node_info, parent_str = self.close.pop(tuple(node))
        # parent=node_info[1]
        parent = eval(parent_str)
        path.append(parent)
        while parent is not None:
            # path.insert(0, node.action)
            node_info, parent_str = self.close.pop(tuple(parent))
            parent = eval(parent_str)
            if parent is not None:
                path.append(parent)
        path.reverse()
        new_path = np.array(path)
        return new_path

class A_star_Node():
    def __init__(self, state, parent=None,  cost=0, h_value=None,f_value=None):

        self.state = state
        self.parent = parent
        self.cost = cost
        self.h_value = h_value
        self.f_value = f_value
