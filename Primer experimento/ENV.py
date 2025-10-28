import time
import torch
import mujoco

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
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    obs, info = env.reset()
    imgs = env.render()
    print(f"Render output type: {type(imgs)}")
    if isinstance(imgs, tuple) and len(imgs) >= 2:
        frame = imgs[1]  # panoramic_view_bgr
    else:
        raise ValueError("env.render() must return a tuple with panoramic view at index 1")

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        #imgs = env.render(options = options)
        imgs = env.render()
        if isinstance(imgs, tuple) and len(imgs) >= 2:
            frame = imgs[1]  # panoramic_view_bgr
        else:
            frame = imgs
        out.write(frame)
        if done:
            print(f"[{env.objeto}] Episode finished at step {step}, reward: {reward}")
            break

    out.release()
    print(f"[{env.objeto}] Video saved to {video_path}")

    if log_mlflow:
        try:
            import mlflow
            mlflow.log_artifact(video_path)
            print(f"[{env.objeto}] Video logged to MLflow: {video_path}")
        except Exception as e:
            print(f"Could not log video to MLflow: {e}")


def get_camera_image(model, data, cam_name, width=400, height=400, options = None):
    # Find the camera ID
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    # Create a renderer



    renderer = mujoco.Renderer(model, height=height, width=width)
    # Render from the given camera
    if options is None:
        renderer.update_scene(data, camera=cam_name)
    else:
        renderer.update_scene(data, cam_name, options)
        model.vis.scale.contactwidth = 0.1
        model.vis.scale.contactheight = 0.03
        model.vis.scale.forcewidth = 0.05
        model.vis.map.force = 0.3
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
    print(f"Extracting s pattern from {input_string}, found: {match.group() if match else 'None'}")

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


class MujocoEnvCNN(gym.Env):
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
        self.n_episodes = 0

        self.force_mean = np.zeros(12)
        self.force_std = np.ones(12)
        self.force_count = 0
        self.predicted_movement = None
        #if scene_path is None:
        #    new_sample = random.sample(os.listdir("/home/diego/TFM/models/scenes/"), 1)[0]
        #    scene_path = new_sample
        self.num_agents = 1
        scene_path = "alarmclock_lift_s4_scene.xml"
        self.scene_path = scene_path

        hand_joint_names = [ "ffj0", "ffj1", "ffj2", "ffj3",
            "mfj0", "mfj1", "mfj2", "mfj3",
            "rfj0", "rfj1", "rfj2", "rfj3",
            "thj0", "thj1", "thj2", "thj3",
            "supination", "wrist_flexion", #comented because these are controlled by the las two components of the predicted vector
        ]
        self.start_time = time.time()
        self.body_parts = {
            #"hand_root": 2,
            #"forearm": 3,
            #"palm": 4,
            "ff_base": 5,
            "ff_proximal": 6,
            "ff_medial": 7,
            "ff_distal": 8,
            "ff_tip": 9,
            "mf_base": 10,
            "mf_proximal": 11,
            "mf_medial": 12,
            "mf_distal": 13,
            "mf_tip": 14,
            "rf_base": 15,
            "rf_proximal": 16,
            "rf_medial": 17,
            "rf_distal": 18,
            "rf_tip": 19,
            "th_base": 20,
            "th_proximal": 21,
            "th_medial": 22,
            "th_distal": 23,
            "th_tip": 24
        }

        self.max_hand_height = 0.1
        self.experiment = 2
        self.alpha = 0.1  # Step on torque adjustments
        self.contact_frames = 0
        self.contact_frames_upward = 0

        self.obs_shape = obs_shape
        self.not_grabbed = True
        self.total_module = 0
        self.summed_total_module = 0
        self.total_reward = 0
        self.reward_success = cfg['reward_success']
        self.penalty_invalid_point = cfg['penalization_out_of_bounds']
        self.frame_idx = 1
        self.x = 0.005
        self.movement_pattern = movement_pattern
        self.max_dist_tol = 0.11
        self.frame_tol = 11
        super().__init__()
        self.done = False
        self.frame_counter = 0
        self.num_frames = num_frames
        self.movement_pattern = movement_pattern
        self.max_hand_height = 0.30
        self.scenes_path = '/'.join(scene_path.split('/')[:-1])
        
        self.scene_results = dict()

        gesture, sub, _ = extract_s_pattern(self.scene_path)

        objeto = gesture.split('_')[0]
        self.objeto = objeto
        objeto_stl = check_object[objeto.replace('_', '')]
        model_path = "/home/diego/TFM/models/scenes/"+gesture+"_"+sub+"_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"Loaded scene {self.scene_path} with object {objeto} from {model_path}")    
        # 1. Ensure simulation is properly initialized
        mujoco.mj_forward(self.model, self.data)


        for _ in range(20):
            mujoco.mj_forward(self.model, self.data)
        self.hand_joint_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in hand_joint_names
        ]

        # 2. Step the simulation a few times


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
        self.mf_distal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mf_distal")
        self.rf_distal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rf_distal")
        self.ff_distal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ff_distal")
        self.th_distal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "th_distal")

        self.grasp_success = self.get_grasp_success()

        self.final_reward = 0

        # For actuators
        #self.action_space = spaces.Box(
        #    low=np.array([self.model.actuator_ctrlrange[i, 0] for i in range(self.model.nu)], dtype=np.float32),
        #    high=np.array([self.model.actuator_ctrlrange[i, 1] for i in range(self.model.nu)], dtype=np.float32),
        #    dtype=np.float32
        #)
        self.action_space = spaces.Box(
                low=np.array([0, 0, 0, 0, -1.4, -1.4] , dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1.4, 1.4], dtype=np.float32),
                shape=(6,),
                dtype=np.float32
                )
        #Get motor instances from the joints:
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                joint_id = self.model.actuator_trnid[i, 0]
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                gain = self.model.actuator_gainprm[i, 0]
                ctrl_range = (self.model.actuator_ctrlrange[i, 0], self.model.actuator_ctrlrange[i, 1])
        

        self.action_space = spaces.Box(
            low=np.array([self.model.actuator_ctrlrange[i, 0] for i in range(self.model.nu)], dtype=np.float32),
            high=np.array([self.model.actuator_ctrlrange[i, 1] for i in range(self.model.nu)], dtype=np.float32),
            dtype=np.float32
        )


        # Observation space: depth image + optionally mask
        # MobileNet feature extractor (pretrained, last pooling layer)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_proprioceptive_dim = self.get_proprioceptive_data().shape[0]
        self.get_task_info_dim = self.get_task_info().shape[0]






        #Creating the shape of the observation space:
        # 12 force components (4 fingers × 3 force components)
        # 12 finger tip positions (4 fingers × 3 position components)
        # 3 object position components (X, Y, Z)
        # 4 target orientation components (quaternion: x, y, z, w)
        # 4 hand orientation components (quaternion: x, y, z, w)
        # 4 hand-to-object orientation components (quaternion: x, y, z, w)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(39,),  # Total of 35 components
            dtype=np.float32
            )


        # Combined observation space
        #self.observation_space = spaces.Dict({
            #'images': image_space,
            #'forces': force_space
        #})

        # Updated observation space without image
        #self.observation_space = spaces.Dict({
            #'forces': force_space,
            #'finger_positions': finger_tip_positions,
            #'object_position': object_position,
            #'target_orientation': target_orientation,
            #'hand_orientation': spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            #'hand_to_object_orientation': spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            #})


        #self._disable_gradients()

        self.object_mask = None


    def contact_hand(self):
        """
        Returns true if the hand is in contact with the object.
        """
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Check if contact is between this tip and the object
            if ((contact.geom1 == self.hand_id and contact.geom2 == self.obj_id) or
                (contact.geom2 == self.hand_id and contact.geom1 == self.obj_id)):
                return True
        return False
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


    def get_proprioceptive_data(self):
        """Return proprioceptive information like joint positions, velocities, etc."""
        # Example: return concatenated array of relevant proprioceptive data
        return np.concatenate([
            self.data.qpos[:15],  # First 7 joint positions
            self.data.qvel[:15],  # First 7 joint velocities
            # Add other relevant proprioceptive data
        ])

    def get_task_info(self):
        """Return task-specific information that might help the critic."""
        # Example: object position, distance to target, etc.
        return np.array([
            self.data.xpos[self.obj_id][0],  # Object x position
            self.data.xpos[self.obj_id][1],  # Object y position
            self.data.xpos[self.obj_id][2],  # Object z position

            #Object velocity:
            self.data.cvel[self.obj_id][0],  # Object x velocity
            self.data.cvel[self.obj_id][1],  # Object y velocity
            self.data.cvel[self.obj_id][2],  # Object z velocity

            # Force applied to the object:

            self.data.cfrc_ext[self.obj_id][0],  # External force x
            self.data.cfrc_ext[self.obj_id][1],  # External force y
            self.data.cfrc_ext[self.obj_id][2],  # External force z
            # Add other task-relevant info, object orientation:
            self.data.xquat[self.obj_id][0],  # Object orientation (quaternion x)
            self.data.xquat[self.obj_id][1],  # Object orientation (quaternion y)
            self.data.xquat[self.obj_id][2],  # Object orientation (quaternion z)
            self.data.xquat[self.obj_id][3],  # Object orientation (quaternion w)

        ])



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
        self.n_episodes += 1
        self.total_reward = 0
        self.contact_frames = 0
        self.contact_frames_upward = 0
        self.start_time = time.time()
        self.total_reward = 0
        self.final_reward = 0
        self.grasp_success = False
        self.total_module = 0
        self.summed_total_module = 0
        info = dict()
        if seed is not None:
            super().reset(seed=seed)
        self.done = False
        self.frame_counter = 0
        self.movement_pattern = 'down'
        self.model = None
        

        # --- Scene sampling based on improvement ---
        scenes_dir = "/home/diego/TFM/models/scenes/"
        all_scenes = os.listdir(scenes_dir)
        epsilon = 0.05  # To avoid division by zero

        # Get rewards for each scene (0 if not attempted)

        #rewards = [self.scene_results.get(extract_s_pattern(scene)[0].split('_')[0], 0.0) for scene  in all_scenes]
        # Compute weights: lower reward, higher probability
        #weights = [1.0 / (r + epsilon) for r in rewards]
        # Normalize weights
        #total_weight = sum(weights)
        #probabilities = [w / total_weight for w in weights]
        counter = 0
        # Sample a scene based on probabilities
        while self.model is None:
            try:
                #sampled_scene = np.random.choice(all_scenes, p=probabilities)

                
                #new_scene_path = random.sample(os.listdir("/home/diego/TFM/models/scenes/"), 1)[0]
                sampled_scene = "cubemedium_pass_1_s7_scene.xml"
                #sampled_scene = "bowl_pass_1_s5_scene.xml"
                sampled_scene = "cylindermedium_inspect_1_s1_scene.xml"
                sampled_scene = "toruslarge_pass_1_s6_scene.xml"
                #sampled_scene = "spheremedium_inspect_1_s3_scene.xml"
                sampled_scene = "stapler_pass_1_s6_scene.xml"
                sampled_scene = "airplane_fly_1_scene.xml"
                sampled_scene = "airplane_fly_1_s1_scene.xml"
                #sampled_scene = "alarmclock_lift_s4_scene.xml"
                sampled_scene = "apple_lift_s6_scene.xml"
                sampled_scene = "banana_lift_s7_scene.xml"
                sampled_scene = "binoculars_pass_1_s6_scene.xml"
                sampled_scene = "doorknob_use_2_s3_scene.xml"
                sampled_scene = "cubemedium_pass_1_s7_scene.xml"
                gesture, sub, _ = extract_s_pattern(sampled_scene)
                
                objeto = gesture.split('_')[0]
                self.objeto = objeto
                objeto_stl = check_object[objeto.replace('_', '')]
                model_path = "/home/diego/TFM/models/scenes/"+gesture+"_"+sub+"_scene.xml"
                self.model = mujoco.MjModel.from_xml_path(model_path)
                self.data = mujoco.MjData(self.model)
            except:
                counter += 1
                if counter > 10:
                    raise RuntimeError("Could not load a valid Mujoco model after 10 attempts.")
                continue
        info['reset'] = f"resetting on scene {sampled_scene} with object {objeto}"
        self.frame_idx = 1
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

        #self.grasp_success = self.get_grasp_success()
        #self.grasp_success = False
        
        # Now, action will be a 4 dimensional vector, in which the corresponding movement is classified. The possible movements are:
        # Power grip
        # Pinch grip
        # Lateral grip
        # Tripod grip
        # The corresponding classification will determine completely the animation followed by the hand.
        #print(f"Action shape: {action.shape}, Action values: {action}")
        if self.experiment == 1:
            clas = np.argmax(action[0:4])
            supination = action[4]
            wrist_flexion = action[5]
            if clas == 0:
                self.predicted_movement = 'power'
            elif clas == 1:
                self.predicted_movement = 'pinch'
            elif clas == 2:
                self.predicted_movement = 'lateral'
            elif clas == 3:
                self.predicted_movement = 'tripod'
       
        elif self.experiment == 2:




            done = False

            object_contact = False
            # Current hand height
            current_hand_pos = self.data.xpos[self.hand_id]
            current_hand_height = current_hand_pos[2]

            # Get current object height
            current_obj_pos = self.data.xpos[self.obj_id]
            current_obj_height = current_obj_pos[2]

            height_diff = abs(current_hand_height - current_obj_height)



            # Movement logic: V pattern movement 'down' -> 'stand' -> 'up'
            if self.movement_pattern == 'down':
                # Moving the hand down until reaching the target height above the object
                # Open the hand while moving down
                if abs(current_hand_height - current_obj_height).sum() > self.max_dist_tol:
                    multiplier = (abs(current_hand_height-current_obj_height).sum() - self.max_dist_tol)**0.3
                    if multiplier < 0:
                        multiplier = 0
                    #Lowering the hand faster if it's far away from the object, slower when it's close
                    self.data.mocap_pos[self.mocap_id] -= [0, 0, self.x*multiplier]
                # If the palm is in contact with the object, change movement pattern to 'stand'
                if self.contact_hand():
                    self.movement_pattern = 'stand'
        
                if abs(current_hand_height - current_obj_height).sum() < self.max_dist_tol:
                        # Change movement pattern to 'up' after reaching the grasping height
                        self.movement_pattern = 'stand'



            if self.movement_pattern == 'stand':
                self.frame_idx += 1
                if abs(current_hand_height - current_obj_height).sum() > self.max_dist_tol:
                    self.movement_pattern = 'down'
                self.frame_counter += 1
                #print(f"frame counter {self.frame_counter}")
                if self.frame_counter > self.frame_tol:
                    self.movement_pattern = 'up'
                if self.force_z_projection() > 0.01:
                    print("z force projection:", self.force_z_projection())
                if self.grasp_success:
                    self.contact_frames += 1
                if self.force_z_projection() > 0.01:
                    self.contact_frames_upward +=1
                    print("Adding upward contact frame:", self.contact_frames_upward)
                    

            if self.movement_pattern == 'up':

                self.frame_idx += 1
                self.grasp_success = self.get_grasp_success()

                if self.grasp_success:
                    self.contact_frames += 1
                if self.force_z_projection() > 0.01:
                    self.contact_frames_upward +=1
                height_diff = abs(current_hand_height - current_obj_height)
                if current_hand_height > self.max_hand_height:
                    # When the hand reaches a certain height, end the episode
                    self.done = True
                    if self.done and self.grasp_success:
                        print(f"[{self.objeto}] SUCCESS - time taken: {time.time() - self.start_time} contact % {self.contact_frames/self.frame_idx :.2f}")
                        print(f"[{self.objeto}] State now is success (reward = 1), previous was {self.scene_results.get(self.objeto, 'not_attempted')}")
                        self.scene_results[self.objeto] = 1
                        self.final_reward = 1.0
                    else:
                        print(f"[{self.objeto}] FAILURE - time taken: {time.time() - self.start_time} contact % {self.contact_frames/self.frame_idx :.2f}")
                        print(f"[{self.objeto}] State now is failure (reward = {self.contact_frames/self.frame_idx :.2f}) , previous was {self.scene_results.get(self.objeto, 'not_attempted')}")
                        if(self.contact_frames/self.frame_idx > self.scene_results.get(self.objeto, 0)):
                            print(f"[{self.objeto}]--> Improved performance on object {self.objeto} from {self.scene_results.get(self.objeto, 0)} to {self.contact_frames/self.frame_idx :.2f}")
                        

                        self.final_reward = 1*self.contact_frames/self.frame_idx + 0*self.contact_frames_upward/self.frame_idx
                        self.scene_results[self.objeto] = self.final_reward

                self.data.mocap_pos[self.mocap_id] += [0, 0, self.x]

            # REWARD CALCULATION
            reward = self.reward()
            contact_dict = self.fingertip_contact()
            
            if (contact_dict[self.ff_tip_id] or contact_dict[self.mf_tip_id] or contact_dict[self.rf_tip_id]) and contact_dict[self.th_tip_id]:
                self.grasp_success = True
            #step the model
            info = dict()
            # Iterating the simulation
            mujoco.mj_step(self.model, self.data)


            #print(f"reward: {reward}")


            # Simulate go up phase, check if object is lifted
            lifted = self.get_lifted_success()


            obs = self._get_obs()




            truncated = False

















            # In this experiment, we input directly the desired position of the fingers on the actuators from the neural network outputs.
            #Action should have 18 dimensions, one for each actuator of the hand.
            assert action.shape[0] == 18, f"Action shape must be (18,), got {action.shape}"
            kp = 50.0  # Your position gain
            kv = 5.0   # Your velocity gain
            for i in range(len(self.hand_joint_indices)):
                if i >= 16:
                    kp = 20.0
                    kv = 2.0
                    self.data.ctrl[i] = action[i]
                    continue
                error = action[i] - self.data.qpos[i]
                self.data.ctrl[i] = kp * error + kv * self.data.qvel[i]  # PD control

            # REWARD CALCULATION
            reward = self.reward()

            #step the model
            info = dict()
            # Iterating the simulation
            mujoco.mj_step(self.model, self.data)

            #print(f"reward: {reward}")


            # Simulate go up phase, check if object is lifted
            lifted = self.get_lifted_success()


            obs = self._get_obs()




            truncated = False

            if self.frame_idx >= self.num_frames:
                pass
                #self.done = True
           

            return obs, reward, self.done, truncated, info

        #self.data.ctrl[:] = action
        

        #print(f"action on step: {action}")


        with torch.no_grad():
            # Initializing variables
            done = False

            object_contact = False
            # Current hand height
            current_hand_pos = self.data.xpos[self.hand_id]
            current_hand_height = current_hand_pos[2]

            # Get current object height
            current_obj_pos = self.data.xpos[self.obj_id]
            current_obj_height = current_obj_pos[2]
            target_positions = np.zeros(len(self.hand_joint_indices))

            # Movement logic: V pattern movement 'down' -> 'stand' -> 'up'
            if self.movement_pattern == 'down':
                # Moving the hand down until reaching the target height above the object
                # Open the hand while moving down
                target_positions = np.array([0.0, 0.0, 0.0, 0.0,   # ffj0-3
                                             0.0, 0.0, 0.0, 0.0,   # mfj0-3
                                             0.0, 0.0, 0.0, 0.0,   # rfj0-3
                                             0.0, 0.0, 0.0, 0.0,   # thj0-3
                                             0.0, 0.0])
                kp = 50.0  # Your position gain
                kv = 5.0   # Your velocity gain
                for i in range(len(self.hand_joint_indices)):
                    error = target_positions[i] - self.data.qpos[i]
                    self.data.ctrl[i] = kp * error + kv * self.data.qvel[i]  # PD control
                if supination != np.nan and wrist_flexion != np.nan:
                    self.data.ctrl[16] = supination 
                    self.data.ctrl[17] = wrist_flexion
                if abs(current_hand_height - current_obj_height).sum() > self.max_dist_tol:
                    multiplier = (abs(current_hand_height-current_obj_height).sum() - self.max_dist_tol)**0.3
                    if multiplier < 0:
                        multiplier = 0
                    #Lowering the hand faster if it's far away from the object, slower when it's close
                    self.data.mocap_pos[self.mocap_id] -= [0, 0, self.x*multiplier]


            if abs(current_hand_height - current_obj_height).sum() < self.max_dist_tol:
                    # Change movement pattern to 'up' after reaching the grasping height
                    self.movement_pattern = 'stand'

            if self.movement_pattern == 'stand' or self.movement_pattern == 'up':
                # On stand, perform the corresponding animation
                
                contact_dict = self.fingertip_contact()
                if self.predicted_movement == 'power' and not self.grasp_success:
                   


                    hand_joint_names = [ "ffj0", "ffj1", "ffj2", "ffj3",
                        "mfj0", "mfj1", "mfj2", "mfj3",
                        "rfj0", "rfj1", "rfj2", "rfj3",
                        "thj0", "thj1", "thj2", "thj3",
                        "supination", "wrist_flexion"
                    ]



                    target_positions = np.array([0.193, 1.53, 1.07, 0.539,   # ffj0-3
                                                    -0.0799, 1.12, 1.27, 0.852,   # mfj0-3
                                                    -0.258, 1.61, 1.2, 0.622,   # rfj0-3
                                                    1.4, 0.675, 0.673, 0.92,   # thj0-3
                                                    -0.168, 0.0])

                    if all(contact_dict.values()):
                        print("All fingertips are in contact with the object. Switching to torque control for stabilization.")
                        adjusted_target_positions = self.data.qpos[self.hand_joint_indices]
                        self.grasp_success = True

                elif self.predicted_movement == 'pinch' and not self.grasp_success:


                    target_positions = np.array([0.0282, 1.02, 0.946, 0.742,   # ffj0-3
                                                 -0.409, -0.196, -0.174, -0.0794,   # mfj0-3
                                                 0.47, 0.111, -0.174, -0.227,   # rfj0-3
                                                 1.38, -0.105, 0.288, 1.02,   # thj0-3
                                                 0.525, -0.021])
                    contact_dict = self.fingertip_contact()
                    if contact_dict[self.ff_tip_id] and contact_dict[self.th_tip_id]:
                        print("Thumb and index fingertips are in contact with the object. Switching to torque control for stabilization.")
                        adjusted_target_positions = self.data.qpos[self.hand_joint_indices]
                        self.grasp_success = True

                elif self.predicted_movement == 'lateral' and not self.grasp_success:

                   
                    target_positions = np.array([0.0047, 0.978, 0.899, 0.603,   # ffj0-3
                                                 0.0893, 0.978, 0.984, 0.622,   # mfj0-3
                                                 0.165, 0.978, 1.1, 0.502,   # rfj0-3
                                                 1.4, 0.409, 0.508, 0.91,   # thj0-3
                                                 0.0, 0.0])

                    #If all the fingers are touching the object, we stop setting these positions, and start with the torque control
                    # in order to stabilize the grasp.
                    contact_dict = self.fingertip_contact()                    
                    if all(contact_dict.values()):
                        print("All fingertips are in contact with the object. Switching to torque control for stabilization.")
                        adjusted_target_positions = self.data.qpos[self.hand_joint_indices]
                        self.grasp_success = True
                    else:
                        adjusted_target_positions = target_positions


                    #target_positions = np.array([0.0, 0.2, 0.2, 0.2,   # ffj0-3
                                                 #0.0, 0.5, 1.0, 1.0,   # mfj0-3
                                                 #0.0, 0.2, 0.2, 0.2,   # rfj0-3
                                                 #0.0, 0.5, 1.0, 1.0,   # thj0-3
                                                 #1.0, 0.0])
                elif self.predicted_movement == 'tripod' and not self.grasp_success:




                    # Pinch grip: close thumb and index, other fingers relaxed
                    target_positions = np.array([-0.258, 0.662, 1.33, 0.723,   # ffj0-3
                                                 -0.409, -0.196, -0.174, -0.227,   # mfj0-3
                                                 0.47, 0.897, 0.862, 0.889,   # rfj0-3
                                                 1.4, 0.269, 0.755, 0.722,   # thj0-3
                                                 0.609, 0.0])
                #setting the ctrl to the corresponding target positions
                    contact_dict = self.fingertip_contact()
                    if contact_dict[self.ff_tip_id] and contact_dict[self.th_tip_id] and contact_dict[self.rf_tip_id]:
                        print(f"Current force projection on Z axis: {self.force_z_projection():.4f}")
                    #Taking into account the torque adjustments calculated to improve grasp stability
                    for i in range(len(self.hand_joint_indices)):
                        torque_adjustments = self.control_forces()
                        #print(f"Torque adjustment for joint {i} ({mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.hand_joint_indices[i])}): {torque_adjustments[i]:.4f}")

                        self.data.ctrl[i] += self.alpha*torque_adjustments[i]


                # Finally, set supination and wrist_flexion
                if supination != np.nan and wrist_flexion != np.nan:
                    self.data.ctrl[16] = supination 
                    #print(f"Supination control value: {self.data.ctrl[-2]}")
                    self.data.ctrl[17] = wrist_flexion
                #print(f"Wrist flexion control value: {self.data.ctrl[-1]}")
                #self.data.ctrl[:] = target_positions
                


                if self.movement_pattern == 'stand':
                    if abs(current_hand_height - current_obj_height).sum() > self.max_dist_tol:
                        self.movement_pattern = 'down'
                    self.frame_counter += 1
                    #print(f"frame counter {self.frame_counter}")
                    if self.frame_counter > self.frame_tol:
                        self.movement_pattern = 'up'
                    


                    #self.done = True
                    #if self.done and self.grasp_success:
                    #    print(f"Episode ended, SUCCESS, time taken: {time.time() - self.start_time}")
                    #else:
                    #    print(f"Episode ended, FAIL, time taken: {time.time() - self.start_time}")
                    # still not going up self.movement_pattern = 'up'

                pass

            if self.movement_pattern == 'up':

                self.grasp_success = self.get_grasp_success()

                height_diff = abs(current_hand_height - current_obj_height)
                if current_hand_height > self.max_hand_height:
                    # When the hand reaches a certain height, end the episode
                    self.done = True
                    if self.done and self.grasp_success:
                        print(f"[{self.objeto}] SUCCESS - time taken: {time.time() - self.start_time} contact % {self.contact_frames/self.frame_idx :.2f}, reward = {self.final_reward:.2f}")
                        print(f"[{self.objeto}] State now is success (reward = 1), previous was {self.scene_results.get(self.objeto, 'not_attempted')}")
                        self.scene_results[self.objeto] = self.total_reward
                        self.final_reward = 1.0
                    else:
                        print(f"[{self.objeto}] FAILURE - time taken: {time.time() - self.start_time} contact % {self.contact_frames/self.frame_idx :.2f}")
                        print(f"[{self.objeto}] State now is failure (reward = {self.contact_frames/self.frame_idx :.2f}) , previous was {self.scene_results.get(self.objeto, 'not_attempted')}")
                        if(self.contact_frames/self.frame_idx > self.scene_results.get(self.objeto, 0)):
                            print(f"[{self.objeto}]--> Improved performance on object {self.objeto} from {self.scene_results.get(self.objeto, 0)} to {self.contact_frames/self.frame_idx :.2f}")
                        

                        self.final_reward = sefl.total_reward #self.contact_frames/self.frame_idx + 0*self.contact_frames_upward/self.frame_idx
                        self.scene_results[self.objeto] = self.total_reward

                self.data.mocap_pos[self.mocap_id] += [0, 0, self.x]

            # REWARD CALCULATION
            reward = self.reward()

            #step the model
            info = dict()
            # Iterating the simulation
            mujoco.mj_step(self.model, self.data)
            self.frame_idx += 1
            if self.grasp_success:
                self.contact_frames += 1
                if self.force_z_projection() > 0.01:
                    self.contact_frames_upward +=1

            #print(f"reward: {reward}")


            # Simulate go up phase, check if object is lifted
            lifted = self.get_lifted_success()


            obs = self._get_obs()




            truncated = False

            if self.frame_idx >= self.num_frames:
                pass
                #self.done = True
           

            return obs, reward, self.done, truncated, info

    def force_z_projection(self):
        fingertip_forces = self.get_fingertip_forces()  # shape (4, 3)
        total_force = np.sum(fingertip_forces, axis=0)  # shape (3,)
        z_projection = np.linalg.norm(total_force[2]) / (np.linalg.norm(total_force) + 1e-6)
        return z_projection


    def fingertip_contact(self):
        contact_count = 0
        fingertip_ids = [self.ff_tip_id, self.mf_tip_id, self.rf_tip_id, self.th_tip_id]
        contact_dict = {tip_id: False for tip_id in fingertip_ids}
        for tip_id in fingertip_ids:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if ((contact.geom1 == tip_id and contact.geom2 == self.obj_id) or
                    (contact.geom2 == tip_id and contact.geom1 == self.obj_id)):
                    contact_dict[tip_id] = True
                    break
        return contact_dict

    def control_forces(self, vertical_axis=np.array([0, 0, 1]), adjustment_strength=0.1):
        """
        Adjust joint torques for contact fingers so the total exerted force on the object
        is as close as possible to the vertical_axis (e.g., upward/z).
        Uses the torque Jacobian for each contact finger.
        """
        # 1. Get which fingertips are touching the object (tip_ids)
        fingertip_ids = [self.ff_tip_id, self.mf_tip_id, self.rf_tip_id, self.th_tip_id]
        contact_fingers = []
        for idx, tip_id in enumerate(fingertip_ids):
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if ((contact.geom1 == tip_id and contact.geom2 == self.obj_id) or
                    (contact.geom2 == tip_id and contact.geom1 == self.obj_id)):
                    contact_fingers.append(tip_id)
                    break

        # 2. Get the forces at each contact finger
        contact_forces = self.get_fingertip_forces()  # shape (num_fingers, 3)

        # 3. Compute current total force
        total_force = np.sum(contact_forces, axis=0)

        # 4. Compute vertical component of total force
        vertical_axis = np.array(vertical_axis) / np.linalg.norm(vertical_axis)
        vertical_component = np.dot(total_force, vertical_axis) * vertical_axis

        # 5. Compute adjustment needed to align total force to vertical
        adjustment = vertical_component - total_force

        # 6. For each contact finger, compute torque adjustment using Jacobian
        # We'll try to distribute the adjustment using the Jacobians of the contact fingers
        # This is a least-squares solution: Find joint torques d_tau such that
        # sum_i J_i^T d_tau_i ≈ adjustment

        # Build stacked Jacobian for all contact fingers
        J_list = []
        for tip_id in contact_fingers:
            # Each J is 3 x n_joints (n_joints = self.model.nv)
            J_pos = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, J_pos, None, tip_id)
            J_list.append(J_pos)

        if not J_list:
            # No contacts, nothing to do
            return np.zeros(self.model.nu)

        # Stack all Jacobians: shape (3, n_joints * n_contacts)
        # To keep it simple, we solve for d_tau for all hand joints together
        # J_stack: (3, n_joints), since all tips affect same joints
        J_stack = np.sum(J_list, axis=0)  # shape (3, n_joints)

        # Least-squares solve: min ||J_stack^T d_tau - adjustment||^2
        # d_tau = pseudo-inverse(J_stack.T) * adjustment
        JT = J_stack.T  # shape (n_joints, 3)
        #print(f"Stacked Jacobian shape: {J_stack.shape}, JT shape: {JT.shape}, adjustment shape: {adjustment.shape}")
        # Compute joint torque adjustment

        d_tau, residuals, rank, s = np.linalg.lstsq(J_stack, adjustment * adjustment_strength, rcond=None)
        # d_tau: shape (n_joints,)

        # 7. Map d_tau to actuators (self.model.nu actuators, self.hand_joint_indices for finger joints)
        # Only update finger joint actuators
        torque_cmd = np.zeros(self.model.nu)
        for i, joint_idx in enumerate(self.hand_joint_indices):
            if i < len(d_tau):
                torque_cmd[i] = d_tau[i]
        
        #Total magnitude of adjustment
        total_adjustment_magnitude = np.linalg.norm(d_tau)
        return torque_cmd



    def check_force(self, tip_id, obj_id, contact, i):
        # Check if contact is between tip and object mesh
        if ((contact.geom1 == tip_id and contact.geom2 == obj_id) or
            (contact.geom2 == tip_id and contact.geom1 == obj_id)):
            force = np.zeros(6)  # 3 force, 3 torque
            mujoco.mj_contactForce(self.model, self.data, i, force)
            force_vector = force[:3]  # Fx, Fy, Fz in world coordinates
            return force_vector, True
        return np.zeros(3), False


    def calculate_force_projection(self):
        """
        Calculate the percentage projection of the total force vector onto the upward direction (Z-axis)
        Returns value between 0 (perpendicular) and 1 (perfectly upward)
        """
        # Get fingertip forces
        fingertip_forces = self.get_fingertip_forces()  # shape (4, 3)
        
        # Sum all force vectors to get resultant force
        total_force = np.sum(fingertip_forces, axis=0)  # shape (3,)

        # Sum all magnitude of forces
        total_magnitude = np.sum(np.linalg.norm(fingertip_forces, axis=1))
        
        # Calculate magnitude of total force
        force_magnitude = np.linalg.norm(total_force)
        
        if force_magnitude < 1e-6:  # Avoid division by zero
            return 0.0
        
        # Normalize the total force vector
        normalized_force = force_magnitude / total_magnitude
        

        
        return normalized_force


    def reward(self):
        reward = 0




        total_force = 0
        final_vector = np.zeros(3)


        # When the movement is up, object must be grabbed, otherwise heavy penalty
        if self.movement_pattern == 'up' and self.grasp_success == False:
            #reward -= 5
            pass
        elif self.movement_pattern == 'stand' and self.grasp_success == False:
            #reward -= 1
            pass
        elif self.movement_pattern == 'up' and self.grasp_success == True:
            #reward += 3
            pass

        force_projection = self.calculate_force_projection()

        # Get force magnitude for additional reward component
        fingertip_forces = self.get_fingertip_forces()
        total_force_magnitude = np.linalg.norm(np.sum(fingertip_forces, axis=0))
        
        #Multiply this reward by each finger in contact with the object:
        contact_dict = self.fingertip_contact()
        contact_multiplier = sum(contact_dict.values())
        fingers_used = sum(contact_dict.values())

        
        # Extra penalty if critical fingers aren't used
        if not contact_dict.get('th', False) and (self.movement_pattern == 'stand' or self.movement_pattern == 'up'):  # No thumb contact
            reward -= 0.5  # Heavy penalty - thumb is crucial for grasping
        
        if fingers_used >= 4 and (self.movement_pattern == 'stand' or self.movement_pattern == 'up'):  # Bonus for using 4+ fingers
            reward += 3


        reward = self.force_z_projection() * contact_multiplier
        
        self.total_reward +=reward # Normalize force magnitude (adjust threshold based on your environment)

        if self.done:
            print(f"Final reward: {self.total_reward:.4f} out of {self.frame_idx} frames")

        return reward



           
    def _get_obs(self):
        # Get the image observations
        imgs, _ = self.render()
        
        # Get the force measurements
        forces = self.get_fingertip_forces().flatten()
        

        #Get also fingertip positions, velocities, joint positions, velocities?
        fingertip_positions = []
        fingertip_velocities = []
        for tip_id in [self.ff_tip_id, self.mf_tip_id, self.rf_tip_id, self.th_tip_id]:
            fingertip_positions.append(self.data.xpos[tip_id])
            fingertip_velocities.append(self.data.cvel[tip_id][:3])
        fingertip_positions = np.array(fingertip_positions).flatten()
        fingertip_velocities = np.array(fingertip_velocities).flatten()
        # Update normalization statistics (optional)
        self._update_force_stats(forces)
        
        # Normalize forces
        normalized_forces = (forces - self.force_mean) / (self.force_std + 1e-8)


        # Get 
        return {
            'images': imgs.astype(np.uint8),
            'forces': normalized_forces.astype(np.float32)
        }

    def _update_force_stats(self, forces):
        # Online mean and std calculation
        self.force_count += 1
        delta = forces - self.force_mean
        self.force_mean += delta / self.force_count
        delta2 = forces - self.force_mean
        self.force_std += delta * delta2

    def list_all_body_parts(self):
        print("=== Mujoco Body Parts ===")
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"Body ID: {i}, Name: {name}")


    def _check_contact(self):
        """Check if fingers are in contact with object"""
        if self.obj_body_id == -1:
            return False
        thumb_flag = False
        other_finger_flag = False
        for contact in self.data.contact:
            # Check if contact involves object and any finger
            if (contact.geom1 == self.obj_body_id or contact.geom2 == self.obj_body_id):
                # Check if the other geometry is a finger
                other_geom = contact.geom2 if contact.geom1 == self.obj_body_id else contact.geom1
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                if geom_name in self.body_parts:
                    #return True #TEMPORARY, SIMPLIFIED
                    if 'th' in geom_name:
                        thumb_flag = True
                    elif 'ff' in geom_name or 'mf' in geom_name or 'rf' in geom_name:
                        other_finger_flag = True
                

        if thumb_flag and other_finger_flag:
            return True
        self.fingers_contact = {'ff': False,
            'mf': False,
            'rf': False,
            'th': False
        }
        force_thumb = None
        forces_finger = []

        contact_id_list = [
            self.ff_tip_id, self.mf_tip_id, self.rf_tip_id, self.th_tip_id,
            self.ff_distal_id, self.mf_distal_id, self.rf_distal_id, self.th_distal_id
            ]

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for contact_id in contact_id_list:
                contact_info = self.check_force(contact_id, self.obj_id, contact, i)
                if contact_info[1] and contact_id in [self.ff_tip_id, self.mf_tip_id, self.rf_tip_id, self.ff_distal_id, self.mf_distal_id, self.rf_distal_id]:
                    forces_finger.append(contact_info[0])
                    object_contact = True
                    if contact_id == self.ff_tip_id:
                        self.fingers_contact['ff'] = True
                    elif contact_id == self.mf_tip_id:
                        self.fingers_contact['mf'] = True
                    elif contact_id == self.rf_tip_id:
                        self.fingers_contact['rf'] = True
                    break
                elif contact_info[1] and contact_id in [self.th_tip_id, self.th_distal_id]:
                    if force_thumb is None:
                        force_thumb = contact_info[0]
                    else:
                        force_thumb += contact_info[0]
                    object_contact = True
                    self.fingers_contact['th'] = True
                    break


        return False


    def check_current_state(self):
        summary = "[FINAL] === Current State Results ===\n"
        print("[FINAL] === Current State Results ===")
        for scene, reward in self.scene_results.items():
            print(f"[FINAL] {scene}: {reward}")
            summary += f"{scene}: {reward:.2f}\n"
        print(f"Average reward: {np.mean(list(self.scene_results.values())) if len(self.scene_results) > 0 else 0}")
        self.current_state = np.mean(list(self.scene_results.values())) if len(self.scene_results) > 0 else 0
        self.std_state = np.std(list(self.scene_results.values())) if len(self.scene_results) > 0 else 0
        summary += f"Average reward: {self.current_state}\n"
        self.summary = summary


    def get_grasp_success(self):
        """
        Simulate the robot moving to grasp point and closing the gripper.
        Returns True if grasp succeeded, False otherwise.
        """
        # Implement Mujoco simulation logic.
        # For now, return True as placeholder.
        #if self.movement_pattern == 'stand' or self.movement_pattern == 'up':

        contact = self._check_contact()
        
        return contact

    def get_lifted_success(self):
        """
        Simulate lifting the object after grasp.
        Returns True if the object is lifted, False otherwise.
        """
        # Implement lifting logic based on Mujoco object position
        # For now, return True as placeholder.
        if self.movement_pattern == 'up':
            contact = self._check_contact()
            return contact
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


    def render(self, options = None):
        # You can implement Mujoco camera rendering here



        img_ru = get_camera_image(self.model, self.data, 'cam_ru_base', width = self.obs_shape[0], height = self.obs_shape[1])
        img_rd = get_camera_image(self.model, self.data, 'cam_rd_base', width = self.obs_shape[0], height = self.obs_shape[1])
        img_lu = get_camera_image(self.model, self.data, 'cam_lu_base', width = self.obs_shape[0], height = self.obs_shape[1])
        img_ld = get_camera_image(self.model, self.data, 'cam_ld_base', width = self.obs_shape[0], height = self.obs_shape[1])
        img_ru_bgr = cv2.cvtColor(img_ru, cv2.COLOR_RGB2BGR)
        img_rd_bgr = cv2.cvtColor(img_rd, cv2.COLOR_RGB2BGR)
        img_lu_bgr = cv2.cvtColor(img_lu, cv2.COLOR_RGB2BGR)
        img_ld_bgr = cv2.cvtColor(img_ld, cv2.COLOR_RGB2BGR)

        panoramic_view = get_camera_image(self.model, self.data, 'panoramic_view', width = self.obs_shape[0], height = self.obs_shape[1], options = options)
        panoramic_view_bgr = cv2.cvtColor(panoramic_view, cv2.COLOR_RGB2BGR)
    
        stacked = np.concatenate(
            [img_lu_bgr[None, ...], 
             img_ld_bgr[None, ...], 
             img_ru_bgr[None, ...], 
             img_rd_bgr[None, ...]
             ], axis=0
        ).transpose(0, 3, 1, 2)



        return stacked, panoramic_view_bgr


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor






class SingleBranchCNN7(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        # Extract the image and force spaces
        image_space = observation_space['images']
        force_space = observation_space['forces']
        
        super().__init__(observation_space, features_dim)
        
        # Image processing CNN (unchanged)
        n_images, n_channels, height, width = image_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(n_images * n_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=5, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(128, 256, kernel_size=5, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_images * n_channels, height, width)
            n_flatten_cnn = self.cnn(sample).shape[1]
        
        # Force processing MLP
        force_dim = force_space.shape[0]
        self.force_mlp = nn.Sequential(
            nn.Linear(force_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Final combined layers
        combined_dim = n_flatten_cnn + 32  # CNN features + force features
        self.combined_linear = nn.Linear(combined_dim, features_dim)
        self.batch_norm = nn.BatchNorm1d(features_dim)

    def forward(self, observations):
        # Split observations into images and forces
        images = observations['images']
        forces = observations['forces']
        
        #Check if images or forces are nan:
        if torch.isnan(images).any():
            print("Warning: NaN values found in images input to CNNExtractor")
        if torch.isnan(forces).any():
            print("Warning: NaN values found in forces input to CNNExtractor")
        batch_size = images.shape[0]
        
        # Process images
        x_images = images.reshape(batch_size, -1, images.shape[-2], images.shape[-1])
        x_images = self.cnn(x_images)
        
        # Process forces
        x_forces = self.force_mlp(forces)
        
        # Combine features
        combined = torch.cat([x_images, x_forces], dim=1)
        #Apply softmax to the first 4 elements of the combined vector (corresponding to the action classification)
        #combined[:, :4] = torch.softmax(combined[:, :4], dim=1
        output = self.batch_norm(self.combined_linear(combined))
        output_clas = torch.softmax(output[:, :4], dim=1)
        
        free_out = torch.tanh(output[:, 4:]) * 1.4  # scale to [-1.4, 1.4]

        #print(f"Output clas should be normalized to sum 1: {output_clas.sum(dim=1)}")
        #assert torch.allclose(output_clas.sum(dim=1), torch.ones_like(output_clas.sum(dim=1))), "Output clas not normalized to sum 1"
        assert output_clas.shape[1] == 4, "Output clas shape incorrect"
        output_rest = free_out
        combined = torch.cat([output_clas, free_out], dim=1)
        #print(f"Combined feature shape: {combined.shape} (should be 6)")
        #print(f"Values out of the network: {combined}")
        return combined
