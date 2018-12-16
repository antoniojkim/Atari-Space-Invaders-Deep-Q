
import gym
import time

from Model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:  ", device)

model = Model().to(device)
model.load_weights("./Attempts/attempt10/model_average")
env = gym.make("SpaceInvaders-v0")

def get_model_move(transformed):
    rewards = model.predict(transformed, device)
    return np.random.choice(np.flatnonzero(rewards == rewards.max()))

def get_random_move(transformed):
    return env.action_space.sample()

def play(get_move, iterations=100, verbose=False):
    cumulative_rewards = []
    for i in range(iterations):
        cumulative_reward = 0
        num_lives_left = 3
        observation = env.reset()
        done = False
        while not done:
            if verbose:
                env.render()
                time.sleep(0.01)

            observation, reward, done, info = env.step(get_move(transform_image(observation)))

            if info["ale.lives"] < num_lives_left:
                num_lives_left -= 1
                # print(f"\r({i} / {iterations})".ljust(15), "Lives: ", num_lives_left, "   Reward: ", str(int(cumulative_reward)).ljust(10), end="")
            elif reward > 0:
                cumulative_reward += reward
                # print(f"\r({i} / {iterations})".ljust(15), "Lives: ", num_lives_left, "   Reward: ", str(int(cumulative_reward)).ljust(10), end="")
            
        cumulative_rewards.append(cumulative_reward)

    mean = np.mean(cumulative_rewards)
    std = np.std(cumulative_rewards, ddof=1)

    print(f"\n{mean}±{std}  :  ({mean-std}, {mean+std})")
    



# 

if __name__ == "__main__":
    play(get_model_move) # Optimal: 158.45±114.6177110963646  :  (43.83228890363539, 273.0677110963646)
    play(get_model_move) # Average: 146.0±105.76799462440746  :  (40.232005375592536, 251.76799462440746)
    play(get_random_move) # 151.2±109.80036981777981  :  (41.39963018222018, 261.0003698177798)