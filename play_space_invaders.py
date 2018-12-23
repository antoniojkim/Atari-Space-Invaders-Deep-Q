
import gym
import time

from SpaceInvadersModel import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:  ", device)

optimal_model = Model("./Attempts/attempt2/max_space_invaders_model").to(device)
# average_model = Model("./Attempts/attempt2/model_average").to(device)
env = gym.make("SpaceInvaders-v0")

def get_optimal_move(transformed):
    rewards = optimal_model.predict(transformed, device)
    return np.random.choice(np.flatnonzero(rewards == rewards.max()))
def get_average_move(transformed):
    rewards = average_model.predict(transformed, device)
    return np.random.choice(np.flatnonzero(rewards == rewards.max()))

def get_random_move(transformed=None):
    return env.action_space.sample()

def play(get_move, iterations=100, verbose=False):
    cumulative_rewards = []
    for i in range(iterations):
        cumulative_reward = 0
        num_lives_left = 3
        observation = env.reset()
        current_state = []
        action = get_random_move()
        done = False
        while not done:
            if verbose:
                env.render()
                time.sleep(0.01)

            if len(current_state) >= 4:
                action = get_move(current_state)
                current_state = []
            else:
                current_state.append(transform_image(observation))

            observation, reward, done, info = env.step(action)

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
    # play(get_model_move) # Optimal: 158.45±114.6177110963646  :  (43.83228890363539, 273.0677110963646)
    # play(get_model_move) # Average: 146.0±105.76799462440746  :  (40.232005375592536, 251.76799462440746)
    # play(get_random_move) # 151.2±109.80036981777981  :  (41.39963018222018, 261.0003698177798)

    # play(get_optimal_move) # 162.95±110.2318378727674  :  (52.71816212723259, 273.1818378727674)
    # play(get_average_move) # 126.7±91.15759469570855  :  (35.54240530429145, 217.85759469570854)
    # play(get_random_move) # 173.25±134.97170485368815  :  (38.27829514631185, 308.22170485368815)

    play(get_optimal_move) # 162.95±110.2318378727674  :  (52.71816212723259, 273.1818378727674)
    # play(get_average_move) # 126.7±91.15759469570855  :  (35.54240530429145, 217.85759469570854)
    play(get_random_move) # 173.25±134.97170485368815  :  (38.27829514631185, 308.22170485368815)