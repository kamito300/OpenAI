import gym
import random
import numpy as np
import chainer
from chainer import Function, Variable, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class Model(Chain):
  def __init__(self):
    super(Model, self).__init__(
      l1 = L.Linear(4, 16),
      l2 = L.Linear(16, 2048),
      l3 = L.Linear(2048, 16),
      l4 = L.Linear(16, 2),
    )

  def __call__(self, x, y):
    return F.mean_squared_error(self.predict(x), y)

  def predict(self, x):
    h1 = F.leaky_relu(self.l1(x))
    h2 = F.leaky_relu(self.l2(h1))
    h3 = F.leaky_relu(self.l3(h2))
    y = F.leaky_relu(self.l4(h3))
    return y


class Agent:
  def __init__(self):
    self.model = Model()
    self.optimizer = optimizers.Adam()
    self.optimizer.setup(self.model)
    self.experience = []
    self.max_experience = 300 * 10
    self.epsilon = 0.99
    self.decay = 0.0005
    self.batch_size = 64 
    self.gamma = 0.9
    self.loss = None 
  

  def predict_action(self, state):
    x = Variable(np.array(state, dtype=np.float32).reshape((1, -1)))
    return self.model.predict(x).data[0]

  def action(self, state):
    action = 0
    if np.random.random() < self.epsilon:
      action = random.randint(0,1)
      self.epsilon -= self.decay
      # print("random action: %f, epsilon: %f" % (action, self.epsilon))
    else:
      action = np.argmax(self.predict_action(state))
      # print("greedy action: %f" % action)
    return action

  def save_experience(self, exp):
    self.experience += exp
    self.experience.sort(key=lambda x:x["total_rewards"])
    exp_size = len(self.experience)
    while exp_size > self.max_experience:
      self.experience.pop(0)
      exp_size = len(self.experience)

    # for i in range(3):
    print("    min exp      : %f" % self.experience[0]["total_rewards"])


    # for i in range(3):
    print("    max exp      : %f" % self.experience[len(self.experience) - 1]["total_rewards"])

  def replay(self):
    if len(self.experience) < self.batch_size:
      return
          
    batch = np.array(random.sample(self.experience, self.batch_size))
    # print(batch)
    x = Variable(np.array(map(lambda x:x["old_state"], batch), dtype=np.float32))
    labels = np.array(self.model.predict(x).data.copy(), dtype=np.float32)
    for i in range(self.batch_size):
      action, reward, new_state, done = batch[i]["action"], batch[i]["reward"], batch[i]["new_state"], batch[i]["done"]
      
      if done:
        labels[i, action] = reward
      else:
        labels[i, action] = reward + self.gamma * np.max(self.predict_action(new_state))

    self.model.zerograds()
    loss = self.model(x, Variable(labels)) 
    self.loss = loss
    loss.backward()
    self.optimizer.update()
      

class Trainer:
  def __init__(self):
    self.agt = Agent()
    self.exp = []
    self.logfile = "result.log"
    self.episode = 0

  def train(self):
    env = gym.make('CartPole-v0')
    # with open(self.logfile, 'w') as f:
    #   f.write('episode, timestamp')
    for i in range(30000):
      print("episode: %d" % i)
      print("    epsilon      : %f" % self.agt.epsilon)
      state = env.reset()
      total_rewards = 0
      self.exp = []
      self.episode = i

      for t in range(300):
        env.render()
        old_state = state.copy()
        action = self.agt.action(state)
        state, reward, done, info = env.step(action)
        total_rewards += reward
        new_state = state.copy()
        self.exp.append({"old_state": old_state, "action": action, "reward": reward, "new_state": new_state, "done": done})
        if done or t == 299:
          for i in range(len(self.exp)):
            self.exp[i]["total_rewards"] = total_rewards
          # with open(self.logfile, 'w') as f:
          #   f.write(str(self.episode) + ',' + str(t) + '\n')
          print("    timestamp    : %d" % t)
          self.agt.save_experience(self.exp)
          break

      self.agt.replay()
      if self.agt.loss is not None:
        print("    loss         : %f" % self.agt.loss.data)
        print("    total rewards: %f" % total_rewards)

trainer = Trainer()
trainer.train()
