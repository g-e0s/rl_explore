import os
import sys
import shutil
from gym import spaces

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from PIL import Image


sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")
from mapgen import Dungeon


class ModifiedDungeon(Dungeon):
    """Use this class to change the behavior of the original env (e.g. remove the trajectory from observation, like here)"""
    def __init__(self,
        width=20,
        height=20,
        max_rooms=3,
        min_room_xy=5,
        max_room_xy=12,
        max_steps: int = 2000
    ):
        observation_size = 11
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size = 11,
            vision_radius = 5,
            max_steps = max_steps
        )

        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 3]) # because we remove trajectory and leave only cell types (UNK, FREE, OCCUPIED)
        self.action_space = spaces.Discrete(3)

    def step(self):
        observation, reward , done, info = super().step()
        observation = observation[:, :, :-1] # remove trajectory
        return observation, reward , done, info
    

if __name__ == "__main__":

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("Dungeon", lambda config: Dungeon(**config))


    CHECKPOINT_ROOT = "tmp/ppo/dungeon"
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = os.getenv("HOME") + "/ray_results1/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = "Dungeon"
    config["env_config"] = {
        "width": 20,
        "height": 20,
        "max_rooms": 3,
        "min_room_xy": 5,
        "max_room_xy": 10,
        "observation_size": 11,
        "vision_radius": 5
    }

    config["model"] = {
        "conv_filters": [
            [16, (3, 3), 2],
            [32, (3, 3), 2],
            [32, (3, 3), 1],
        ],
        "post_fcnet_hiddens": [32],
        "post_fcnet_activation": "relu",
        "vf_share_layers": False,
    }


    config["rollout_fragment_length"] = 100
    config["entropy_coeff"] = 0.1
    config["lambda"] = 0.95
    config["vf_loss_coeff"] = 1.0



    agent = ppo.PPOTrainer(config)


    N_ITER = 500
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

    #env = Dungeon(50, 50, 3)

    for n in range(N_ITER):
        result = agent.train()
        #print(result.keys())
        file_name = agent.save(CHECKPOINT_ROOT)

        print(s.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
        ))

        # sample trajectory
        if (n+1)%5 == 0:
            env = Dungeon(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
            obs = env.reset()
            Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).save('tmp.png')

            frames = []

            for _ in range(500):
                action = agent.compute_single_action(obs)

                frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
                frames.append(frame)

                #frame.save('tmp1.png')
                obs, reward, done, info = env.step(action)
                if done:
                    break

            frames[0].save(f"out.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000/60)