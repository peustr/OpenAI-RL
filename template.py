import gym


env = gym.make("Boxing-ram-v0")

for i_episode in range(100):
    print("Episode:", i_episode + 1)
    state = env.reset()
    done = False
    timestep = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(timestep + 1))
            break
        timestep += 1

env.close()
