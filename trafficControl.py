import traci
from collections import defaultdict
import numpy as np
import math 
from actor_critic_torch import Agent
import json
import time

sumoBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "chromepet.sumocfg"]
traci.start(sumoCmd)

step = 1 



class State:
    def __init__(self, tl_id, curr_phase, controlled_lanes, lane_waiting_time, lane_vehicle_count):
        self.tl_id = tl_id
        self.phase = curr_phase
        self.controlled_lanes = controlled_lanes
        self.lane_waiting_time = defaultdict(int)
        self.lane_vehicle_count = defaultdict(int)
        self.average_waiting_time = 1 
        self.vehicle_count = 1
        for lane_id in controlled_lanes:
            self.lane_waiting_time[lane_id] = lane_waiting_time[lane_id]
            self.average_waiting_time += lane_waiting_time[lane_id]
            self.lane_vehicle_count[lane_id] = lane_vehicle_count[lane_id]
            self.vehicle_count += lane_vehicle_count[lane_id]
        self.average_waiting_time /= self.vehicle_count 

hasAgent = defaultdict(bool)
agent = {}

def observe(curr_state,tl_id):
    observation = [curr_state[tl_id].phase,curr_state[tl_id].average_waiting_time,curr_state[tl_id].vehicle_count]
    traffic_light_ids = traci.trafficlight.getIDList()
    for tl_id in traffic_light_ids:
        pos = traci.junction.getPosition(tl_id)
        for lane in curr_state[tl_id].controlled_lanes:
            observation.append(lane_waiting_time[lane])
            observation.append(lane_vehicle_count[lane])
        for neighbour in traffic_light_ids:
            if neighbour == tl_id:
                continue
            neighbour_pos = traci.junction.getPosition(neighbour)
            factor = (((pos[0]-neighbour_pos[0])**2+(pos[1]-neighbour_pos[1])**2)**0.5)/500
            observation[1] += curr_state[neighbour].average_waiting_time * (0.98 ** factor)
            observation[2] += curr_state[neighbour].vehicle_count * (0.98 ** factor)
    return observation

def performAction(curr_state,tl_id):
    observation = observe (curr_state,tl_id)    
    if (hasAgent[tl_id] == False):
        agent[tl_id] = Agent(alpha=0.0003, beta=0.0003, reward_scale=2,env_id=tl_id,input_dims=[len(observation)], tau=0.005, batch_size=256, layer1_size=256, layer2_size=256,n_actions=2)
        hasAgent[tl_id] = True
    return agent[tl_id].choose_action(observation),observation

def getReward(reward,tl_id):
    total_reward = reward[tl_id]
    traffic_light_ids = traci.trafficlight.getIDList()
    print(traffic_light_ids)
    for tl_id in traffic_light_ids:
        pos = traci.junction.getPosition(tl_id)
        for neighbour in traffic_light_ids:
            if neighbour == tl_id:
                continue
            neighbour_pos = traci.junction.getPosition(neighbour)
            factor = (((pos[0]-neighbour_pos[0])**2+(pos[1]-neighbour_pos[1])**2)**0.5)/500
            total_reward += reward[neighbour] * (0.98 ** factor)
    return total_reward


def softmax(x):
    exp_x = np.exp(abs(x))
    return exp_x / exp_x.sum(axis=0)



while step < 2000:
    
    curr_state = {}
    curr_observation = {}
    action = defaultdict(int)
    vehicle_ids = traci.vehicle.getIDList()
    lane_vehicle_count = defaultdict(int)
    lane_waiting_time = defaultdict(int)

    for v_id in vehicle_ids:
        lane = traci.vehicle.getLaneID(v_id)
        vType = traci.vehicle.getVehicleClass(v_id)
        factor = 1
        if vType == "emergency":
            factor = 1000
        lane_vehicle_count[lane] += 1
        lane_waiting_time[lane] += factor * traci.vehicle.getWaitingTime(v_id)

    traffic_light_ids = traci.trafficlight.getIDList()
    
    for tl_id in traffic_light_ids:
        id = tl_id
        curr_phase = traci.trafficlight.getPhase(tl_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        curr_state[tl_id] = State(tl_id,curr_phase,controlled_lanes,lane_waiting_time,lane_vehicle_count)
    
    for tl_id in traffic_light_ids:
        action[tl_id],curr_observation[tl_id] = performAction(curr_state,tl_id)
        arr = np.array(action[tl_id])
        probs = softmax(arr)
        phase = np.random.choice(len(arr), p=probs)
        traci.trafficlight.setPhase(tl_id,phase*2)


    traci.simulationStep()
    
    new_state = {}
    new_observation = {}
    reward = {}
    vehicle_ids = traci.vehicle.getIDList()
    
    
    lane_vehicle_count = defaultdict(int)
    lane_waiting_time = defaultdict(int)

    emergency_average = 1
    emergency_count = 1
    vehicle_count = 1
    vechicle_average = 1

    for v_id in vehicle_ids:
        lane = traci.vehicle.getLaneID(v_id)
        vType = traci.vehicle.getVehicleClass(v_id)
        factor = 1
        if vType == "emergency":
            factor = 1000
            emergency_average += traci.vehicle.getWaitingTime(v_id)
            emergency_count += 1
        else :
            vehicle_count += 1
            vechicle_average += traci.vehicle.getWaitingTime(v_id)
        
        lane_vehicle_count[lane] += 1
        lane_waiting_time[lane] += factor * traci.vehicle.getWaitingTime(v_id)
    

    traffic_light_ids = traci.trafficlight.getIDList()
    for tl_id in traffic_light_ids:
        id = tl_id
        curr_phase = traci.trafficlight.getPhase(tl_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        new_state[tl_id] = State(tl_id,curr_phase,controlled_lanes,lane_waiting_time,lane_vehicle_count)
    
    for tl_id in traffic_light_ids:
        average_waiting_time = 0
        variance = 0
        for lane in curr_state[tl_id].controlled_lanes:
            average_waiting_time += (1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane]);
        for lane in curr_state[tl_id].controlled_lanes:
            lane_average = (1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane])
            variance += (lane_average - average_waiting_time/12)**2
        variance /= 12
        reward[tl_id] = - average_waiting_time**2 - variance
    
    for tl_id in traffic_light_ids:
        new_observation[tl_id] = observe(new_state,tl_id)
        total_reward = getReward(reward,tl_id)
        agent[tl_id].remember(curr_observation[tl_id], action[tl_id], total_reward, new_observation[tl_id], step%5==0)
        if (step%10==0): 
            agent[tl_id].learn()
        print("Reward",total_reward)

    
    step += 1

d = {}
d['e_avg'] = emergency_average/emergency_count
d['v_avg'] = vechicle_average/vehicle_count

with open('multi.json', 'w') as f:
    json.dump(d, f)

traci.close()

