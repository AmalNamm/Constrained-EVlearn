import sys
import pickle
import os
sys.path.append("C:\\Users\\Tiago Fonseca\\Documents\\GitHub\\CityLearn")
from citylearn.agents.EVs.maddpg import MADDPG as RLAgent
import time

# Later on, or in another script:
# agent = MADDPG(env)
# Load the model
loaded_agent = RLAgent.from_saved_model("maddpg_trained.pth")

def load_from_pickle(file_name):
    data = []
    with open(file_name, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:  # This error will be raised when we reach the end of the file.
                break
    return data

stored_data = load_from_pickle('method_calls.pkl')
individual_runtimes_predict = []
for timestep_array in stored_data:
    print(timestep_array)
    start_time = time.time()  # Get the current time
    actions = loaded_agent.predict_deterministic(timestep_array)
    end_time = time.time()  # Get the current time again after the function has run
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    individual_runtimes_predict.append(elapsed_time)
print(sum(individual_runtimes_predict) / len(individual_runtimes_predict))