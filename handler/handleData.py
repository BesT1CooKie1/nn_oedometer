# Data preparation for Oedometer class
import torch

def generate_data(oedo:dict, oedo_class, sigma_t:list, amount_trainings_data:int):
    list_output = []
    list_input = []
    for i in range(amount_trainings_data):
        oedo["sigma_t"] = sigma_t[i]
        oedo_output = oedo_class(**oedo)
        list_output.append(oedo_output.e_s[0])
        list_input.append(oedo_output.sigma_t[0])
    tensor_input, tensor_output = create_tensor(list_input, list_output)
    return list_input, list_output, tensor_input, tensor_output

def create_tensor(list_input, list_output):
    tensor_input = torch.tensor(list_input, dtype=torch.float).unsqueeze(-1)
    tensor_output = torch.tensor(list_output, dtype=torch.float).unsqueeze(-1)
    return tensor_input, tensor_output