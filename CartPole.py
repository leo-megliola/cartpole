import numpy as np
import gym                      #importing gym and numpy

def discretafier(state, buckets):
    answer = []
    for i in range(len(state)):
        for j in range(len(buckets[i])):
            if state[i] < buckets[i][j]:
                answer.append(j)
                break
        else:
            answer.append(j)
    return answer


env = gym.make('CartPole-v0')
state = env.reset()

z = 5
b0 = [2.0*float(x)/z for x in range(-z,z,1)]
b1 = [12.0*float(x)/z for x in range(-z,z,1)]
b2 = [float(x)/z for x in range(-z,z,1)]
b3 = [float(x)/z for x in range(-z,z,1)]
print(b0,b1,b2,b3)
'''
b0 = [-1.0,-0.5,0.0,0.5,1.0]
b1 = [-1.0,-0.5,0.0,0.5,1.0]
b2 = [-9.0,-6.0,-3.0,0.0,3.0,6.0,9.0]
b3 = [-1.0,-0.5,0.0,0.5,1.0]
print('====>',state, discretafier(state,[b0,b1,b2,b3]))
'''
done = False
explore = 0.01
alpha = 0.1
gamma = 1.0
steps = 0
moves = 0

q = np.zeros((len(b0)+1,len(b1)+1,len(b2)+1,len(b3)+1,2))
q = np.random.rand(len(b0)+1,len(b1)+1,len(b2)+1,len(b3)+1,2)
for n in range(10000000):
    obs = env.reset()
    ds = discretafier(obs,[b0,b1,b2,b3])
    done = False
    __ = 0
    while not done:
        moves += 1
        __ += 1
        if np.random.random() < explore:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[ds[0],ds[1],ds[2],ds[3]])
        obs, reward, done, _ = env.step(action)
        #print(obs)
        do = discretafier(obs,[b0,b1,b2,b3])
        q[ds[0],ds[1],ds[2],ds[3],action] = (1-alpha) * q[ds[0],ds[1],ds[2],ds[3],action] + alpha * (reward + gamma * np.max(q[do[0],do[1],do[2],do[3]]))
        ds = do
        steps += 1
        if n % 1000 == 0:
            env.render()
    if n % 1000 == 0:
        print(n, float(moves)/1000.0)
        moves = 0
print('steps =',steps/(n+1))
