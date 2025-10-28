
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
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

import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ENV import *


import torch.nn as nn
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_checker import check_env


# To:
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


import torch.nn as nn


from sb3_contrib import RecurrentPPO

from ENV import MujocoEnvCNN as MujocoEnv
#from ENV import CNNExtractor

from custom_callback import EpochCheckpointCallback, ResetTrackingCallback, SafeResetTrackingCallback, send_text, send_video, MLflowGradAndRewardCallback

TELEGRAM_TOKEN = 'ZZZ'
TELEGRAM_CHAT_ID ='ZZZ'

import cv2


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
   
    
    
    

model_save_dir = "./model_checkpoints"
img_save_dir = "./panoramic_views"



callback = MLflowGradAndRewardCallback()


# Use in training:


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


# Create environment instance
env = MujocoEnv(cfg=cfg
                #, obs_shape = (300, 300, 7)
                )
check_env(env)





checkpoint_dir = "./checkpoints_rppo"

# Regex to extract timesteps and mean reward from filename
pattern = re.compile(r"rppo_model_(\d+)_reward_([-\d\.]+)\.zip")

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
print(f"Number of actions: {n_actions}")
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))



import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # Define separate feature extractors or MLPs if needed

    def extract_features(self, obs):
        return obs['policy_obs'], obs['critic_obs']

    def forward(self, obs, deterministic=False):
        policy_obs, critic_obs = self.extract_features(obs)
        # Actor: compute action from policy_obs
        action = self.actor(policy_obs)
        # Critic: compute value from critic_obs
        value = self.critic(critic_obs)
        return action, value

from sb3_contrib import RecurrentPPO

try:
# Find the checkpoint with the best reward
    best_checkpoint = max(checkpoints, key=lambda x: x[2])
    best_fname, best_timesteps, best_reward = best_checkpoint

# Load the best model
    best_model_path = os.path.join(checkpoint_dir, best_fname)
    model = RecurrentPPO.load(best_model_path)
    #model = RecurrentPPO.load(best_model_path)
    model.set_env(env)  # <-- Add this line!

    print(f"Loaded best model: {best_fname} with reward {best_reward}")
except:
    print(f"Couldn't load checkpoint, starting training from scratch")
    

    from sb3_contrib import RecurrentPPO

    policy_kwargs = dict(
        net_arch=[512, 512, 256],  # Deeper network for complex coordination
        activation_fn=nn.ReLU,
        lstm_hidden_size=256,  # Larger LSTM for temporal dependencies
        n_lstm_layers=2,
    )


    model = RecurrentPPO(
        "MlpLstmPolicy",  # or "MultiInputLstmPolicy" for dict obs
        env,
        learning_rate=3e-4,
    #    buffer_size=10_000,
        batch_size=256,
    #    tau=0.005,
        gamma=0.99,
    #    train_freq=1,
    #    gradient_steps=1,
        verbose=1,
        #policy_kwargs={"n_lstm_layers": 2, "lstm_hidden_size": 512},
        #policy_kwargs=policy_kwargs,
        tensorboard_log="./rppo_grab_tensorboard/",
        device='cuda' if torch.cuda.is_available() else 'cpu',
    #    action_noise=action_noise
    )

    #model = RecurrentPPO("MlpLstmPolicy", v, verbose=1)

# Manual training with periodic evaluation
total_timesteps = 1000000
checkpoint_interval = 100
current_timesteps = 0
print(f"Starting training")
mean_reward  = 0
os.makedirs("./checkpoints_rppo", exist_ok=True)
os.makedirs("./videos_rppo", exist_ok=True)

mlflow.start_run()
while current_timesteps < total_timesteps:
    env.reset()
    send_text(f"[Recurrent - PPO - {env.objeto}] Starting training segment at {current_timesteps} timesteps")
    print(f"[{env.objeto}] Recording a video of the agent")

    record_final_video(model, env, video_path = f"./videos_rppo/rppo_model_{current_timesteps}.mp4")
    send_video(f"./videos_rppo/rppo_model_{current_timesteps}.mp4", caption = f"RPPO_current_timesteps_{current_timesteps}_reward_{env.final_reward:.2f}_sac")    
    # Train for a segment
    #print(f"Recording a video of the agent")
    #record_final_video(model, env, video_path = f"./videos/model_{current_timesteps}_reward_{mean_reward:.2f}.mp4")
    #send_video(f"./videos/model_{current_timesteps}_reward_{mean_reward:.2f}.mp4")    
    timesteps_to_train = min(checkpoint_interval, total_timesteps - current_timesteps)
    print(f"Training on {timesteps_to_train} on RPPO") 
    model.learn(total_timesteps=timesteps_to_train, reset_num_timesteps=False, callback = callback)
    current_timesteps += timesteps_to_train
    
    # Evaluate after each segment (safe because training is paused)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    mlflow.log_metric("periodic_mean_reward", mean_reward, step=current_timesteps)
    mlflow.log_metric("std_reward", std_reward, step=current_timesteps)
    mlflow.log_metric("reward", env.final_reward, step=current_timesteps)
    mlflow.log_metric("n_episodes", env.n_episodes, step=current_timesteps)

    env.check_current_state()
    send_text(env.summary)
    if env.current_state > 0.85:
        env.max_hand_height += 0.02
        print(f"[{env.objeto}] Current success rate: {env.current_state}, increasing max_hand_height to {self.max_hand_height}")
        send_text(f"[{env.objeto}] Current success rate: {env.current_state}, increasing max_hand_height to {self.max_hand_height}", TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        env.current_state = 0.0  # Reset current state to avoid multiple increments in one go
        env.scene_results = dict()  # Restart the training with these weights but a new final hand
    mlflow.log_metric("current success rate", env.current_state, step=current_timesteps)
    send_text(f"Current success rate: {env.current_state}")
    # Save checkpoint
    try:
        model.save(f"./checkpoints_rppo/rppo_model_{current_timesteps}_reward_{env.final_reward:.2f}.zip")
        mlflow.log_artifact(f"./checkpoints_rppo/rppo_model_{current_timesteps}_reward_{env.final_reward:.2f}.zip")
    except Exception as e:
        print(f"Error saving model: {e}")
        send_text(f"Error saving model: {e}")
    print(f"[{env.objeto}] Checkpoint at {current_timesteps} timesteps, Mean reward: {env.final_reward:.2f}.")
    print(f"[{env.objeto}] Recording a video of the agent")
    record_final_video(model, env, video_path = f"./videos_rppo/rppo_model_{current_timesteps}.mp4")
    send_video(f"./videos_rppo/rppo_model_{current_timesteps}.mp4", caption = f"current_timesteps_{current_timesteps}_reward_{env.final_reward:.2f}_sac")    
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


