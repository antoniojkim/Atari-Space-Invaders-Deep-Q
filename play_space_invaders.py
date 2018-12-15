
import gym
import time

from Model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:  ", device)

model = Model().to(device)
model.load_weights("./Attempts/attempt8/model_max_cumulative_reward")
env = gym.make("SpaceInvaders-v0")

for _ in range(3):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.01)

        transformed = transform_image(observation)
        rewards = model.predict(transformed, device)
        action = np.random.choice(np.flatnonzero(rewards == rewards.max()))

        observation, reward, done, info = env.step(action)

# max 560

time.sleep(10)