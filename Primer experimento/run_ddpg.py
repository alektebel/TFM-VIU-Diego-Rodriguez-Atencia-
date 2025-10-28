
import mujoco.viewer
import cv2
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import yaml
import random
import re
import mlflow
# To:
import gymnasium as gym
from gymnasium import spaces

import torchvision.transforms as T
from torchvision.models import mobilenet_v2

import cv2

def record_final_video(model, env, video_path="final_agent.mp4", max_steps=200, fps=20, log_mlflow=True):
    """
    Record a video of the trained agent using the panoramic view from the environment.
    """
    obs, info = env.reset()
    imgs = env.render()
    if isinstance(imgs, tuple) and len(imgs) >= 3:
        frame = imgs[2]  # panoramic_view_bgr
    else:
        raise ValueError("env.render() must return a tuple with panoramic view at index 2")

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        imgs = env.render()
        if isinstance(imgs, tuple) and len(imgs) >= 3:
            frame = imgs[2]  # panoramic_view_bgr
        else:
            frame = imgs
        out.write(frame)
        if done:
            print(f"Episode finished at step {step}, reward: {reward}")
            break

    out.release()
    print(f"Video saved to {video_path}")

    if log_mlflow:
        try:
            import mlflow
            mlflow.log_artifact(video_path)
            print(f"Video logged to MLflow: {video_path}")
        except Exception as e:
            print(f"Could not log video to MLflow: {e}")


def get_camera_image(model, data, cam_name, width=400, height=400):
    # Find the camera ID
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    # Create a renderer
    renderer = mujoco.Renderer(model, height=height, width=width)
    # Render from the given camera
    renderer.update_scene(data, camera=cam_name)
    rgb_array = renderer.render()
    return rgb_array  # shape (height, width, 3), dtype=uint8


def extract_s_pattern(input_string):
    """
    Search for s[0-9]+ pattern (s followed by one or more digits)
    """
    # Pattern matches 's' followed by one or more digits
    pattern = r's\d+'  # + means one or more digits

    # Search for the pattern
    match = re.search(pattern, input_string)

    if match:
        # Extract the matched pattern
        s_pattern = match.group()

        # Split the string into before and after
        before = input_string[:match.start()]
        after = input_string[match.end():]

        return before[:-1], s_pattern, after
    else:
        return None, None, None

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

scenes_path = cfg['scenes_dir']

scenes = os.listdir(scenes_path)


stl_objects = [
"airplane.stl", "alarm_clock.stl", "apple.stl", "banana.stl", "binoculars.stl",
"bowl.stl", "camera.stl", "phone.stl", "cube_large.stl", "cube_medium.stl",
"cube_small.stl", "cup.stl", "cylinder_large.stl", "cylinder_medium.stl",
"cylinder_small.stl", "door_knob.stl", "elephant.stl", "eyeglasses.stl",
"flashlight.stl", "flute.stl", "hammer.stl",  "headphones.stl",
"knife.stl", "light_bulb.stl", "mouse.stl", "mug.stl", "pan.stl", "piggy_bank.stl",
"ps_controller.stl", "pyramid_large.stl", "pyramid_medium.stl", "pyramid_small.stl",
"rubber_duck.stl", "scissors.stl", "sphere_large.stl", "sphere_medium.stl",
"sphere_small.stl", "stanford_bunny.stl", "stapler.stl", "toothbrush.stl",
"toothpaste.stl", "torus_large.stl", "torus_medium.stl", "torus_small.stl",
"train.stl", "utah_teapot.stl", "water_bottle.stl", "wine_glass.stl", "wristwatch.stl",
]

check_object = dict()

for obj in stl_objects:
    check_object[obj.replace('.stl', '').replace('_', '')] = obj
    

class MujocoEnv(gym.Env):
    """
    Mujoco environment for GG-CNN robotic grasping.
    - Receives penalty if the selected grasp point is not on the object.
    - Receives penalty if the lifting phase does not result in the object being lifted.
    - Optionally supports an object mask for efficient grasp validation.
    """

    def __init__(self, scene_path=None, movement_pattern = 'down', obs_shape = (300, 300, 3), num_frames = 1000000, cfg=None):
        """
        Args:
            mj_scene: Initialized Mujoco scene/model (mujoco_py.MjSim or equivalent).
            object_mask: [H, W] binary mask of object location in the camera view (optional).
            reward_success: Reward for successful grasp.
            penalty_invalid_point: Penalty if selected grasp point is not on the object.
            penalty_failed_lift: Penalty if the object is not lifted after grasp.
        """
        
        
        if scene_path is None:
            new_sample = random.sample(os.listdir("/home/diego/TFM/models/scenes/"), 1)[0]
            scene_path = new_sample
            
        
        self.obs_shape = obs_shape
        self.not_grabbed = True
        self.total_module = 0
        self.summed_total_module = 0
        self.total_reward = 0
        self.reward_success = cfg['reward_success']
        self.penalty_invalid_point = cfg['penalization_out_of_bounds']
        self.frame_idx = 0
        self.x = 0.005
        self.movement_pattern = movement_pattern
        self.max_dist_tol = 0.07
        self.frame_tol = 50
        super().__init__()
        self.done = False
        self.frame_counter = 0
        self.num_frames = num_frames
        self.movement_pattern = movement_pattern
        self.max_hand_height = 0.30
        self.scenes_path = '/'.join(scene_path.split('/')[:-1])
        
        if scene_path is None:
            new_sample = random.sample(os.listdir("/home/diego/TFM/models/scenes/"))
        self.scene_path = scene_path.split('/')[-1]
    
        gesture, sub, _ = extract_s_pattern(self.scene_path)
        
        objeto = gesture.split('_')[0]
        objeto_stl = check_object[objeto.replace('_', '')]
        model_path = "/home/diego/TFM/models/scenes/"+gesture+"_"+sub+"_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
                # 1. Ensure simulation is properly initialized
        mujoco.mj_forward(self.model, self.data)

        # 2. Step the simulation a few times
        for _ in range(20):
        
            mujoco.mj_step(self.model, self.data)
        joint_ranges = {}
        
        self.hand_root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
        self.obj_body_id = self._find_object_body()

        tip_body_ids = []
        tip_body_names = []
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.obj_body_id)
            if name and "3" in name:
                tip_body_ids.append(body_id)
                tip_body_names.append(name)
        #print("Tip body IDs:", tip_body_ids)
        #print("Tip body names:", tip_body_names)
        
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            #print(f"Body ID: {body_id}, Name: {name}")
        



        
        
        self.mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_root_mocap")
        
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                joint_id = self.model.actuator_trnid[i, 0]
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                gain = self.model.actuator_gainprm[i, 0]
                ctrl_range = (self.model.actuator_ctrlrange[i, 0], self.model.actuator_ctrlrange[i, 1])
                joint_ranges[joint_name] = ctrl_range

        # Get all body names using mj_id2name
        body_names = []
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name:
                body_names.append(name)


        # DEBUG: Print all actuators and their properties
        #print("=== ACTUATOR DEBUG INFO ===")
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                joint_id = self.model.actuator_trnid[i, 0]
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                gain = self.model.actuator_gainprm[i, 0]
                ctrl_range = (self.model.actuator_ctrlrange[i, 0], self.model.actuator_ctrlrange[i, 1])
                #print(f"Actuator {i}: {actuator_name} -> Joint: {joint_name}")
                #print(f"  Gain: {gain}, Ctrl range: {ctrl_range}")


        # Get body IDs correctly
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
        self.obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, objeto_stl.replace('.stl', '')+"_mesh")

        self.ff_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ff_tip")
        self.mf_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mf_tip")
        self.rf_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rf_tip")
        self.th_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "th_tip")


        
        # For actuators
        self.action_space = spaces.Box(
            low=np.array([self.model.actuator_ctrlrange[i, 0] for i in range(self.model.nu)], dtype=np.float32),
            high=np.array([self.model.actuator_ctrlrange[i, 1] for i in range(self.model.nu)], dtype=np.float32),
            dtype=np.float32
        )
        
        
        # Observation space: depth image + optionally mask
        # MobileNet feature extractor (pretrained, last pooling layer)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mob = mobilenet_v2(pretrained=True).to(self.device)
        mob.eval()
        self.mobilenet = mob.features   # Remove classifier, use pooled features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Determine MobileNet output dim
        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, 224, 224).to(self.device)
            mob_features = self.mobilenet(dummy_img)
            pooled = self.pool(mob_features)
            mobilenet_dim = pooled.view(1, -1).shape[1]

        # Fingertips: ff, mf, rf, th
        self.num_fingers = 4
        self.force_dim = 3

        # Observation space: [mobilenet_features, distance, forces]
        obs_dim = mobilenet_dim + 1 + self.num_fingers * self.force_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
       

        #self._disable_gradients()

        self.object_mask = None


    def get_fingertip_forces(self):
        """
        Returns a list of force vectors (Fx, Fy, Fz) for each fingertip.
        """
        force_vectors = []
        fingertip_ids = [self.ff_tip_id, self.mf_tip_id, self.rf_tip_id, self.th_tip_id]  # Add more if needed
        for tip_id in fingertip_ids:
            tip_force = np.zeros(3)
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                # Check if contact is between this tip and the object
                if ((contact.geom1 == tip_id and contact.geom2 == self.obj_id) or
                    (contact.geom2 == tip_id and contact.geom1 == self.obj_id)):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    tip_force += force[:3]  # sum forces if multiple contacts
            force_vectors.append(tip_force)
        return np.array(force_vectors)  # shape (num_fingers, 3)


    def extract_mobilenet_features(self, img):
        """
        img: np.ndarray (H, W, 3), dtype=uint8
        Returns: np.ndarray of size mobilenet_dim
        """
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.mobilenet(img_tensor)
            pooled = self.pool(features)
            mobilenet_vector = pooled.view(-1).cpu().numpy()
        return mobilenet_vector

    def get_fingertip_forces(self):
        """Returns a list of force vectors (Fx, Fy, Fz) for each fingertip."""
        force_vectors = []
        fingertip_ids = [self.ff_tip_id, self.mf_tip_id, self.rf_tip_id, self.th_tip_id]
        for tip_id in fingertip_ids:
            tip_force = np.zeros(3)
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if ((contact.geom1 == tip_id and contact.geom2 == self.obj_id) or
                    (contact.geom2 == tip_id and contact.geom1 == self.obj_id)):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    tip_force += force[:3]
            force_vectors.append(tip_force)
        return np.array(force_vectors)  # shape (num_fingers, 3)


    def _find_object_body(self):
        """Find the object body in the scene"""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and any(keyword in body_name.lower() for keyword in ['object', 'mesh', 'geom']):
                return i
        return -1
        
        
    def reset(self, seed=None):
        """
        Resets the environment and returns:
        obs: 1D feature vector
        info: dict (can contain episode metadata)
        """
        self.total_reward = 0
        self.total_module = 0
        self.summed_total_module = 0
        info = dict()
        if seed is not None:
            super().reset(seed=seed)
        self.done = False
        self.frame_counter = 0
        self.movement_pattern = 'down'
        self.max_hand_height = 0.5
        self.model = None
        while self.model is None:
            try:
                new_scene_path = random.sample(os.listdir("/home/diego/TFM/models/scenes/"), 1)[0]
                gesture, sub, _ = extract_s_pattern(new_scene_path)
                objeto = gesture.split('_')[0]
                objeto_stl = check_object[objeto.replace('_', '')]
                model_path = "/home/diego/TFM/models/scenes/"+gesture+"_"+sub+"_scene.xml"
                self.model = mujoco.MjModel.from_xml_path(model_path)
                self.data = mujoco.MjData(self.model)
            except:
                continue
        mujoco.mj_forward(self.model, self.data)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        print(f"Training on scene {new_scene_path}")  
        info['reset'] = f"resetting on scene {new_scene_path} with object {objeto}"
        self.frame_idx = 0
        self.done = False
        obs = self._get_obs()
        return obs, info

    def _disable_gradients(self):
        """Ensure environment doesn't create computation graphs"""
        torch.set_grad_enabled(False)

    def step(self, action):
        """
        Args:
            action: grasp parameters.
        Returns:
            obs, reward, done, info/
        """


        self.data.ctrl[:] = action
        #print(f"action on step: {action}")
        

        with torch.no_grad():

            reward = 0
            #print(f"Current frame: {self.frame_idx}, movement_pattern {self.movement_pattern}")

            # Get current heights
        
            current_hand_pos = self.data.xpos[self.hand_id]
            current_hand_height = current_hand_pos[2]
            
            #print(f"hand_pos: {current_hand_pos}")
            #print(f"hand_height: {current_hand_height}")
            
            
            current_obj_pos = self.data.xpos[self.obj_id]
            current_obj_height = current_obj_pos[2]
            
            #print(f"obj_pos: {current_obj_pos}")
            #print(f"obj_height: {current_obj_height}")
            done = False        
            object_contact = False        
            forces_finger = []
            force_thumb = None
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                # Check if contact is between tip and object mesh
                if ((contact.geom1 == self.ff_tip_id and contact.geom2 == self.obj_id) or
                    (contact.geom2 == self.ff_tip_id and contact.geom1 == self.obj_id)):
                    force = np.zeros(6)  # 3 force, 3 torque
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    force_vector = force[:3]  # Fx, Fy, Fz in world coordinates
                    forces_finger.append(force_vector)
                    object_contact = True
                    # If you expect only one contact, break here
                if ((contact.geom1 == self.mf_tip_id and contact.geom2 == self.obj_id) or
                    (contact.geom2 == self.mf_tip_id and contact.geom1 == self.obj_id)):
                    force = np.zeros(6)  # 3 force, 3 torque
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    force_vector = force[:3]  # Fx, Fy, Fz in world coordinates
                    forces_finger.append(force_vector)
                    object_contact = True

                    # If you expect only one contact, break here
                if ((contact.geom1 == self.rf_tip_id and contact.geom2 == self.obj_id) or
                    (contact.geom2 == self.rf_tip_id and contact.geom1 == self.obj_id)):
                    force = np.zeros(6)  # 3 force, 3 torque
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    force_vector = force[:3]  # Fx, Fy, Fz in world coordinates
                    forces_finger.append(force_vector)
                    object_contact = True

                    # If you expect only one contact, break here
                if ((contact.geom1 == self.th_tip_id and contact.geom2 == self.obj_id) or
                    (contact.geom2 == self.th_tip_id and contact.geom1 == self.obj_id)):
                    force = np.zeros(6)  # 3 force, 3 torque
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    force_thumb = force[:3]  # Fx, Fy, Fz in world coordinates
                     
                    object_contact = True
                    reward += 1
                    # If you expect only one contact, break here
                
            total_force = 0
            final_vector = np.zeros(3)
            
            for vec in forces_finger:
                total_force += np.sqrt(vec.dot(vec))
                final_vector += vec
            #Rewarding when the agent uses the thumb to grab things
            
            grabbing_module = 0
            if force_thumb is not None and final_vector is not None:
                thumb = np.sqrt(force_thumb.dot(force_thumb))
                fingers = np.sqrt(final_vector.dot(final_vector))
                print(f"force thumb {force_thumb}")
                print(f"final vector {final_vector}")
                print(f"reward before {reward}")
                print(f" force thumb module {np.sqrt(force_thumb.dot(force_thumb))}")
                print(f" final vector module {np.sqrt(final_vector.dot(final_vector))}")
                if thumb > 0.001 and fingers > 0.001:
                    resulting_force = force_thumb+final_vector
                    grabbing_module = np.sqrt(resulting_force.dot(resulting_force))
                    print(f"grabbing_module {grabbing_module}")
            if grabbing_module > 0.001 :
                    print(f"force thumb {force_thumb}")
                    print(f"final vector {final_vector}")
                    print(f"reward before {reward}")
                    print(f" force thumb module {np.sqrt(force_thumb.dot(force_thumb))}")
                    print(f" final vector module {np.sqrt(final_vector.dot(final_vector))}")
                    

                    reward = reward + (np.sqrt(force_thumb.dot(force_thumb)) + np.sqrt(final_vector.dot(final_vector)) / grabbing_module)
                    print(f"reward after {reward}")
            
            # Penalizing when grabbing things without the thumb

            module_final_vector = np.sqrt(final_vector.dot(final_vector))
            grasp_penalty = 0
            if self.movement_pattern == 'down':
                
                if abs(current_hand_height - current_obj_height).sum() > self.max_dist_tol:
                    multiplier = (abs(current_hand_height-current_obj_height).sum() - self.max_dist_tol)**0.3
                    if multiplier < 0:
                        multiplier = 0
                    self.data.mocap_pos[self.mocap_id] -= [0, 0, self.x*multiplier]
                #print(f"current object height: {self.data.xpos[self.obj_id][2] }")
                #print(f"current hand height {current_hand_height}")
                #print(f"height tolerance, {self.max_dist_tol}")
                if abs(current_hand_height - current_obj_height).sum() < self.max_dist_tol:
                    self.movement_pattern = 'stand'


            if object_contact == False and self.movement_pattern == 'stand':
                #print(f"negative reward no contact")
                reward -= 0.1
            elif object_contact == True:
                reward += 0.01


                #print(f"current hand height: {self.data.xpos[self.hand_id][2] }")
            
            if self.movement_pattern == 'up':
                
                
                height_diff = abs(current_hand_height - current_obj_height)
                # Apply penalty proportional to the difference
                grasp_penalty = -height_diff
                if height_diff > 1:
                    self.not_grabbed = True
                if current_hand_height > self.max_hand_height:
                    done = True
                    print(f"Episode ended, final reward {self.total_reward}, object not grabbed: {self.not_grabbed}, total force aplied: {self.total_module}, summed total force applied: {self.summed_total_module}")
                self.data.mocap_pos[self.mocap_id] += [0, 0, self.x]
            if self.movement_pattern == 'stand':

                self.frame_counter += 1
               
                # Only explore every few frames to avoid excessive randomness
                if self.frame_counter % 50 == 0 and 1 == 0:  # Explore every 5 frames
                    print(f"Exploring grab types")
                    exploration_type = random.choice([
                        'full_close', 'partial_close', 'finger_wave', 
                        'thumb_only', 'pincer', 'random'
                    ])
                        
                    if exploration_type == 'full_close':
                        # Close all fingers completely
                        for i in range(self.model.nu):
                            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            if actuator_name and any(finger in actuator_name for finger in ['ff', 'mf', 'rf', 'th']):
                                self.data.ctrl[i] = self.model.actuator_ctrlrange[i, 1]  # Max close
                        
                    elif exploration_type == 'partial_close':
                        # Close fingers partially (50-80%)
                        for i in range(self.model.nu):
                            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            if actuator_name and any(finger in actuator_name for finger in ['ff', 'mf', 'rf', 'th']):
                                low, high = self.model.actuator_ctrlrange[i]
                                close_amount = low + random.uniform(0.5, 0.8) * (high - low)
                                self.data.ctrl[i] = close_amount
                        
                    elif exploration_type == 'finger_wave':
                        # Sequential finger closing
                        fingers = ['ff', 'mf', 'rf', 'th']
                        finger_to_close = fingers[self.frame_counter % len(fingers)]
                        for i in range(self.model.nu):
                            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            if actuator_name and finger_to_close in actuator_name:
                                self.data.ctrl[i] = self.model.actuator_ctrlrange[i, 1]  # Close this finger
                        
                    elif exploration_type == 'thumb_only':
                        # Only close thumb
                        for i in range(self.model.nu):
                            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            if actuator_name and 'th' in actuator_name:
                                self.data.ctrl[i] = self.model.actuator_ctrlrange[i, 1]  # Close thumb
                            elif actuator_name and any(finger in actuator_name for finger in ['ff', 'mf', 'rf']):
                                self.data.ctrl[i] = self.model.actuator_ctrlrange[i, 0]  # Open other fingers
                        
                    elif exploration_type == 'pincer':
                        # Pincer grasp (thumb + index finger)
                        for i in range(self.model.nu):
                            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            if actuator_name and ('th' in actuator_name or 'ff' in actuator_name):
                                self.data.ctrl[i] = self.model.actuator_ctrlrange[i, 1]  # Close thumb and index
                            elif actuator_name and any(finger in actuator_name for finger in ['mf', 'rf']):
                                self.data.ctrl[i] = self.model.actuator_ctrlrange[i, 0]  # Open middle and ring
                        
                    elif exploration_type == 'random':
                        # Random finger positions
                        for i in range(self.model.nu):
                            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                            if actuator_name and any(finger in actuator_name for finger in ['ff', 'mf', 'rf', 'th']):
                                low, high = self.model.actuator_ctrlrange[i]
                                self.data.ctrl[i] = random.uniform(low, high)
                    
                    # Small reward for exploration
                    reward += 0.01


                #print(f"frame counter {self.frame_counter}")
                if self.frame_counter > self.frame_tol:
                    self.movement_pattern = 'up'
                pass
            
            if grasp_penalty < 0:
                reward += grasp_penalty

            # If object heigh
            
            
            #step the model 
            info = dict()
            mujoco.mj_step(self.model, self.data)
            self.frame_idx += 1
            
            #print(f"reward: {reward}")
            self.total_reward += reward
          
            #print(f"total reward: {self.total_reward}")

            #print(f"total force: {total_force}")
            #print(f"module_final_vector: {module_final_vector}")
            if force_thumb is not None and final_vector is not None:
                self.total_module += (np.sqrt(force_thumb.dot(force_thumb)) + np.sqrt(final_vector.dot(final_vector)))
            self.summed_total_module += grabbing_module
            # Penalty if grasp point is not on object mask
            if self.object_mask is not None:
                if self.object_mask[u, v] == 0:
                    # Invalid grasp point
                    obs = self._get_obs()
                    reward = self.penalty_invalid_point
                    done = True
                    info['reason'] = 'invalid_grasp_point'
                    return obs, reward, done, info

            # Simulate grasp in Mujoco
            grasp_success = self.get_grasp_success()

            # Simulate go up phase, check if object is lifted
            lifted = self.get_lifted_success()


            obs = self._get_obs()
            
            
            
            
            truncated = False
                    
            if self.frame_idx >= self.num_frames:
                pass
                #self.done = True
            
            return obs, reward, done, truncated, info

    def render(self):
        # You can implement Mujoco camera rendering here
        
        img_ff = get_camera_image(self.model, self.data, 'cam_ff_base', width = self.obs_shape[0], height = self.obs_shape[1])
        img_mf = get_camera_image(self.model, self.data, 'cam_mf_base', width = self.obs_shape[0], height = self.obs_shape[1])

        img_ff_bgr = cv2.cvtColor(img_ff, cv2.COLOR_RGB2BGR)
        img_mf_bgr = cv2.cvtColor(img_mf, cv2.COLOR_RGB2BGR)


        panoramic_view = get_camera_image(self.model, self.data, 'panoramic_view', width = self.obs_shape[0], height = self.obs_shape[1])
        panoramic_view_bgr = cv2.cvtColor(panoramic_view, cv2.COLOR_RGB2BGR)


        
        
        return img_ff_bgr, img_mf_bgr, panoramic_view_bgr


    def _get_obs(self):
        """
        Returns a 1D feature vector:
        [MobileNet features, distance to object, fingertip forces]
        """
        img_ff, img_mf, _ = self.render()
        mobilenet_features = self.extract_mobilenet_features(img_ff)  # shape: (mobilenet_dim,)
        hand_pos = self.data.xpos[self.hand_id]
        obj_pos = self.data.xpos[self.obj_id]
        dist = np.linalg.norm(hand_pos - obj_pos)
        fingertip_forces = self.get_fingertip_forces().flatten()  # shape: (num_fingers * 3,)
        obs = np.concatenate([mobilenet_features, [dist], fingertip_forces])
        return obs.astype(np.float32)


    def get_grasp_success(self):
        """
        Simulate the robot moving to grasp point and closing the gripper.
        Returns True if grasp succeeded, False otherwise.
        """
        # Implement Mujoco simulation logic.
        # For now, return True as placeholder.
        return False

    def get_lifted_success(self):
        """
        Simulate lifting the object after grasp.
        Returns True if the object is lifted, False otherwise.
        """
        # Implement lifting logic based on Mujoco object position
        # For now, return True as placeholder.
        return False

    def set_object_mask(self, mask):
        """
        Dynamically set or update the object mask.
        """
        self.object_mask = mask

        
    def _set_hand_position(self, new_position):
        """Set hand position through free joint"""
        # Find free joint for hand_root
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                body_id = self.model.jnt_bodyid[i]
                if body_id == self.hand_root_id:
                    qpos_adr = self.model.jnt_qposadr[i]
                    self.data.qpos[qpos_adr:qpos_adr+3] = new_position
                    break
                    
    def _check_contact(self):
        """Check if fingers are in contact with object"""
        if self.obj_body_id == -1:
            return False
        
        for contact in self.data.contact:
            # Check if contact involves object and any finger
            if (contact.geom1 == self.obj_body_id or contact.geom2 == self.obj_body_id):
                # Check if the other geometry is a finger
                other_geom = contact.geom2 if contact.geom1 == self.obj_body_id else contact.geom1
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                
                if geom_name and any(finger in geom_name for finger in ["ff", "mf", "rf", "th"]):
                    return True
        
        return False
    
    
    
    
import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ENV import *



import torch.nn as nn
class GraspMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class SingleBranchCNN7(BaseFeaturesExtractor):
    """
    Single-branch CNN for 7-channel input
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            #sample = sample.permute(0, 3, 1, 2)  # [batch, 7, H, W]
            dim = self.cnn(sample).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:


        x = self.cnn(observations)
        return self.fc(x)

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env


# To:
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback



from custom_callback import EpochCheckpointCallback, ResetTrackingCallback, SafeResetTrackingCallback, send_video, MLflowGradAndRewardCallback

TELEGRAM_TOKEN = 'ZZZ'
TELEGRAM_CHAT_ID ='ZZZ'


model_save_dir = "./model_checkpoints"
img_save_dir = "./panoramic_views"



# Use in training:


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


import torch.nn as nn




# Create environment instance
env = MujocoEnv(cfg=cfg
                #, obs_shape = (300, 300, 7)
                )
check_env(env)

callback = EpochCheckpointCallback(
    env,
    model_save_dir=model_save_dir,
    img_save_dir=img_save_dir,
    telegram_token=TELEGRAM_TOKEN,
    telegram_chat_id=TELEGRAM_CHAT_ID,
    verbose=1
)


callback = MLflowGradAndRewardCallback()

model_save_dir = "./model_checkpoints"
img_save_dir = "./panoramic_images"


# Custom policy for 9-channel input
policy_kwargs = dict(
    activation_fn=nn.ReLU,
    net_arch=dict(pi=[256, 128], vf=[256, 128]),  
    features_extractor_class=SingleBranchCNN7,  
    #features_extractor_class = FrameStackCNN,
    features_extractor_kwargs=dict(features_dim=256)
)




# Create PPO agent




import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy




checkpoint_dir = "./checkpoints_ddpg"

# Regex to extract timesteps and mean reward from filename
pattern = re.compile(r"ddpg_model_(\d+)_reward_([-\d\.]+)\.zip")

# Gather all checkpoints and their rewards
checkpoints = []
for fname in os.listdir(checkpoint_dir):
    match = pattern.match(fname)
    if match:
        timesteps = int(match.group(1))
        reward = float(match.group(2))
        checkpoints.append((fname, timesteps, reward))


print(checkpoints)
from stable_baselines3.common.noise import NormalActionNoise

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))



try:
# Find the checkpoint with the best reward
    best_checkpoint = max(checkpoints, key=lambda x: x[2])
    best_fname, best_timesteps, best_reward = best_checkpoint

# Load the best model
    best_model_path = os.path.join(checkpoint_dir, best_fname)
    model = DDPG.load(best_model_path)
    model.set_env(env)  # <-- Add this line!

    print(f"Loaded best model: {best_fname} with reward {best_reward}")
except:
    print(f"Couldn't load checkpoint, starting training from scratch")
    


    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=1e-3,             # Typical: 1e-3 to 1e-4
        buffer_size=1000000,            # Large enough for diverse experience
        learning_starts=10000,          # Wait before starting updates
        batch_size=256,                 # 128-256 is common
        tau=0.005,                      # Soft update target net, 0.005 default
        gamma=0.99,                     # Discount factor
        train_freq=(1, "step"),         # Train every step
        gradient_steps=1,               # Number of update steps per train_freq
        action_noise=action_noise,              # Add noise for exploration (see below)
        policy_kwargs=policy_kwargs,    # Your custom policy/network
        verbose=1,
        tensorboard_log="./ddpg_grab_tensorboard/",
        device="cpu",                   # Use "cuda" if you have a GPU
    )

# Manual training with periodic evaluation
total_timesteps = 100000
checkpoint_interval = 10000
current_timesteps = 0
print(f"Starting training")
mean_reward  = 0


mlflow.start_run()
mlflow.log_metric("test", 1, step = 0)
while current_timesteps < total_timesteps:
    print(f"Recording a video of the agent")
    record_final_video(model, env, video_path = f"./videos_ddpg/ddpg_model_{current_timesteps}_reward_{mean_reward:.2f}.mp4")
    send_video(f"./videos_ddpg/ddpg_model_{current_timesteps}_reward_{mean_reward:.2f}.mp4", caption = f"current_timesteps_{current_timesteps}_reward_{mean_reward:.2f}_ddpg")    
    # Train for a segment
    #print(f"Recording a video of the agent")
    #record_final_video(model, env, video_path = f"./videos/model_{current_timesteps}_reward_{mean_reward:.2f}.mp4")
    #send_video(f"./videos/model_{current_timesteps}_reward_{mean_reward:.2f}.mp4")    
    
    timesteps_to_train = min(checkpoint_interval, total_timesteps - current_timesteps)
    print(f"Training on {timesteps_to_train}") 
    model.learn(total_timesteps=timesteps_to_train, reset_num_timesteps=False, callback = callback)
    current_timesteps += timesteps_to_train
    
    # Evaluate after each segment (safe because training is paused)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    mlflow.log_metric("periodic_mean_reward", mean_reward, step=current_timesteps)
    mlflow.log_metric("std_reward", std_reward, step=current_timesteps)


    # Save checkpoint
    model.save(f"./checkpoints_ddpg/ddpg_model_{current_timesteps}_reward_{mean_reward:.2f}.zip")
    mlflow.log_artifact(f"./checkpoints_ddpg/ddpg_model_{current_timesteps}_reward_{mean_reward:.2f}.zip")
    
    print(f"Checkpoint at {current_timesteps} timesteps, Mean reward: {mean_reward:.2f}.")
    print(f"Recording a video of the agent")
    record_final_video(model, env, video_path = f"./videos_ddpg/ddpg_model_{current_timesteps}_reward_{mean_reward:.2f}.mp4")
    send_video(f"./videos_ddpg/ddpg_model_{current_timesteps}_reward_{mean_reward:.2f}.mp4", caption = f"current_timesteps_{current_timesteps}_reward_{mean_reward:.2f}_ddpg")    
# Final evaluation and video recording
print("Training completed. Recording final video...")
record_final_video(model, env)

mlflow.end_run()

#mlflow.start_run()
# Train!
#model.learn(total_timesteps=1_400_000, callback = callback)
#mlflow.end_run()


# Save trained agent
model.save("grasping_policy_9channel")

