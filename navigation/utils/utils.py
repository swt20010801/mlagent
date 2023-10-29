import cv2
import numpy as np
import torch
def persp2pano(img,fov=np.pi/2,size=1000):
    width =img.shape[1]
    height=img.shape[0]
    
    
    lon=np.arange(size)/size*fov-fov/2
    lat=np.arange(size)/size*fov-fov/2
    lon,lat=np.meshgrid(lon,lat)
    # print(lon.shape,lat.shape)
    R=128
    x=R*np.cos(lat)*np.sin(lon)
    y=-R*np.sin(lat)
    z=R*np.cos(lat)*np.cos(lon)

    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

    xyz=xyz/np.expand_dims(z,axis=-1) 
    # print(xyz)

    f=width/2/np.tan(fov/2)
    XY=(xyz*f)[:,:,:2].astype(np.float32)
    XY[:,:,0]=XY[:,:,0]+width/2
    XY[:,:,1]=-XY[:,:,1]+height/2

    return cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

def image_2_pano(image,fov=np.pi/2,image_size=256):
    """
    image : Tensor(4,3,h,w)
    """
    image=image.permute(0,2,3,1).cpu().numpy()


    image_0=image[0,:,:,:]
    image_1=image[1,:,:,:]
    image_2=image[2,:,:,:]
    image_3=image[3,:,:,:]

    pano=[]
    pano.append(persp2pano(image_0,fov,image_size))
    pano.append(persp2pano(image_1,fov,image_size))
    pano.append(persp2pano(image_2,fov,image_size))
    pano.append(persp2pano(image_3,fov,image_size))
    image=np.concatenate(pano,axis=1)
    #pano的最左边（首位）对应着机器人正面的朝向
    image=np.concatenate([image[:,image_size//2:,:],image[:,:image_size//2,:],],axis=1)
    
    cv2.imwrite("pano.jpg",image*255.)

    # image=image/255.
    image -= (0.485, 0.456, 0.406)
    image /= (0.229, 0.224, 0.225)

    image=np.transpose(image, (2, 0, 1))
    image=torch.from_numpy(image).unsqueeze(0)
    return image