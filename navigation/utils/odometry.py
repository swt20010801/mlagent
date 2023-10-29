import numpy as np
import torch

def odometry(last_position_x, last_position_y, last_roatation,now_position_x, now_position_y, now_rotation):
    distance=torch.sqrt((last_position_x-now_position_x)**2+(last_position_y-now_position_y)**2)
    theta=now_rotation-last_roatation

    distance=distance.unsqueeze(0)
    theta=theta.unsqueeze(0)

    theta += 5*torch.pi/180 * torch.randn(1)
    distance += 0.05 * torch.randn(1)

    return distance,theta