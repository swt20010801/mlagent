from typing import Callable, List, Dict, Tuple, Optional, Union, Any
import abc

from mlagents.torch_utils import torch, nn
import cv2
from mlagents.trainers.torch_entities.agent_action import AgentAction
import numpy as np

#######
from mlagents.navigation.utils.utils import image_2_pano
from mlagents.navigation.utils.odometry import odometry
from mlagents.navigation.utils.CANN_loc import CANN_loc


from mlagents.navigation.CANN.hd_cell import HD_CELL
from mlagents.navigation.CANN.pose_cell import POSE_CELL
from mlagents.navigation.model.fplocnet import FpLocNet
from mlagents.navigation.dataset.unity_dataset import unityDataset
from mlagents.navigation.eval_utils import sample_floorplan
from torch.utils.data import Dataset, DataLoader

def loc_plan_act(
    inputs: List[torch.Tensor],
) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
    
    device=torch.device("cuda")
    if_new_episode=inputs[4][0][0]
    now_position_x=inputs[4][0][1]
    now_position_y=inputs[4][0][2]
    now_rotation  =inputs[4][0][3]/180*np.pi


    
    if True:#if_new_episode==1:
        house_id=11
        cfg = {
            "Q": 100,
            "Q_refine": 20,
            "D": 128,
            "G": 32,
            "H": 32,
            "dist_max": 10,
            "Vr": 64,
            "V": 16,
            "disable_semantics": False,
            "disable_pointnet": False,
            "fov": 360 ,
            "view_type": "eview",
        }
        vis_model_path="ml-agents/mlagents/navigation/models/try_wo_scheduler.pth"
        dataset_dir="ml-agents/mlagents/navigation/unitydataset"
        eval_dataset=unityDataset(dataset_dir=dataset_dir,
                                is_training=False,n_sample_points=2048,testing_set=[house_id])
        eval_dataloader=DataLoader(dataset=eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )

        model = FpLocNet(cfg).to(device=device)
        model.load_state_dict(torch.load(vis_model_path))
        data=eval_dataset[0]
        for eval_data in eval_dataloader:
            for k in eval_data.keys():
                if torch.is_tensor(eval_data[k]) and not eval_data[k].is_cuda:
                    eval_data[k] = eval_data[k].cuda()

            if cfg["disable_semantics"]:
                eval_data["bases_feat"][..., -2:] = 0
            sample_ret = sample_floorplan(
                eval_data,
                model,
                cfg,
                sample_grid=0.1,
                batch_size=256,
            )
            samples_loc=sample_ret["samples_loc"]
            rot_samples = torch.arange(16).float() / 16 * 360

            grid_shape  = sample_ret["samples_original_shape"]
            grid_coords = sample_ret["samples_loc"].reshape((grid_shape[0],grid_shape[1],2))
            grid_feat   = sample_ret["samples_feat"]
            break
        ###初始化###
        ###选择初始点周围的点作为初始化备选点，此处0.1为范围，这个值很小，可以改的大一点降低初始化位置的精确度
        near_particles=[]
        init_activation_loc=[]
        num_acivate_init=10
        for i in range(grid_coords.shape[0]):
            for j in range(grid_coords.shape[1]):
                if(abs(grid_coords[i,j,0]-now_position_x)<0.1 and abs(grid_coords[i,j,1]-now_position_y)<0.1):
                    # initial_activation.append([i,j])
                    near_particles.append([i,j])
        rand = np.random.rand
        args_coords = np.arange(len(near_particles))
        selected_args = np.random.choice(args_coords, num_acivate_init)

        for i in range(num_acivate_init):
            x = near_particles[selected_args[i]][0]
            y = near_particles[selected_args[i]][1]
            init_activation_loc.append([x, y])
        ###初始化pose cells和head cells
        pose_cell=POSE_CELL(GC_X_DIM=grid_shape[0],GC_Y_DIM=grid_shape[1],init_activation_loc=init_activation_loc)
        hd_cell=HD_CELL(init_activation_yaw=now_rotation)

        last_position_x=now_position_x
        last_position_y=now_position_y
        last_roatation=now_rotation
    # else:
    #     last_position_x=FF['last_position_x']
    #     last_position_y=FF['last_position_y']
    #     last_roatation =FF['last_roatation']
    #     model=FF['model']
    #     grid_feat=FF['grid_feat']
    #     samples_loc=FF['samples_loc']
    #     rot_samples=FF['rot_samples']
    #     grid_shape=FF['grid_shape']
    #     pose_cell=FF['pose_cell']
    #     hd_cell=FF['hd_cell']
    #     grid_coords=FF['grid_coords']

    distance,theta=odometry(last_position_x, last_position_y, last_roatation,now_position_x, now_position_y, now_rotation)
    images_from_4directions=torch.cat((inputs[0],inputs[1],inputs[2],inputs[3]),dim=0).permute(0,3,1,2)
    pano_image=image_2_pano(images_from_4directions)

    now_loc_est,now_rot_est=CANN_loc(pano_image,device,model,grid_feat,samples_loc,rot_samples,grid_shape,pose_cell,hd_cell,grid_coords,distance,theta)



    # if if_new_episode==1.:
    #     FF={
    #     'last_position_x':now_position_x,
    #     'last_position_y':now_position_y,
    #     'last_roatation' :now_rotation,
    #     'model':model,
    #     'grid_feat':grid_feat,
    #     'samples_loc':samples_loc,
    #     'rot_samples':rot_samples,
    #     'grid_shape':grid_shape,
    #     'pose_cell':pose_cell,
    #     'hd_cell':hd_cell,
    #     'grid_coords':grid_coords,
    #     }
    # else:
    #     FF['last_position_x']=now_position_x
    #     FF['last_position_y']=now_position_y
    #     FF['last_roatation']=now_rotation
    #     FF['model']=FF['model']
    #     FF['grid_feat']=FF['grid_feat']
    #     FF['samples_loc']=FF['samples_loc']
    #     FF['rot_samples']=FF['rot_samples']
    #     FF['grid_shape']=FF['grid_shape']
    #     FF['pose_cell']=pose_cell
    #     FF['hd_cell']=hd_cell
    #     FF['grid_coords']=FF['grid_coords']

    action=AgentAction(None,[torch.ones((1,1))])

    return action