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
      l2 = L.Linear(16, 256),
      l3 = L.Linear(256, 512),
      l4 = L.Linear(512, 1024),
      l5 = L.Linear(1024, 2),
    )

  def __call__(self, x, y):
    return F.mean_squared_error(self.predict(x), y)

  def predict(self, x):
    h1 = F.leaky_relu(self.l1(x))
    h2 = F.leaky_relu(self.l2(h1))
    h3 = F.leaky_relu(self.l3(h2))
    h4 = F.leaky_relu(self.l4(h3))
    y = F.leaky_relu(self.l5(h4))
    return y

class Agent:
  def __init__(self):
    self.model = Model()
    self.optimizer = optimizers.Adam()
    self.optimizer.setup(self.model)
    self.experience = []
    self.max_experience = 300 * 100
    self.epsilon = 0.5
    self.decay = 0.99
    self.batch_size = 32
    self.gamma = 0.9

  def predict_action(self, state):
    x = Variable(np.array(state, dtype=np.float32).reshape((1, -1)))
    return self.model.predict(x).data[0]

  def action(self, state):
    action = 0
    if np.random.random() < self.epsilon:
      action = random.randint(0,1)
      self.epsilon = self.decay * self.epsilon
    else:
      action = np.argmax(self.predict_action(state))
    return action

  def save_experience(self, old_state, action, reward, new_state):
    self.experience.append({"old_state": old_state, "action": action, "reward": reward, "new_state": new_state})
    self.experience.sort(key=lambda x:x["reward"])
    if len(self.experience) > self.max_experience:
      self.experience.pop

  def replay(self):
    if len(self.experience) < self.batch_size:
      return
          
    batch = np.array(random.sample(self.experience, self.batch_size))
    x = Variable(np.array(map(lambda x:x["old_state"], batch), dtype=np.float32))
    labels = np.array(self.model.predict(x).data.copy(), dtype=np.float32)
    for i in range(self.batch_size):
      action, reward, new_state = batch[i]["action"], batch[i]["reward"], batch[i]["new_state"]
      labels[i, action] = reward + self.gamma * np.max(self.predict_action(new_state))

    self.model.zerograds()
    loss = self.model(x, Variable(labels)) 
    loss.backward()
    self.optimizer.update()
      

class Trainer:
  def __init__(self):
    self.agt = Agent()

  def train(self):
    env = gym.make('CartPole-v0')
    for i in range(30000):
      print("episode: %d" % i)
      state = env.reset()

      for t in range(300):
        env.render()
        old_state = state.copy()
        action = self.agt.action(state)
        state, reward, done, info = env.step(action)
        if done:
          reward = -200 
        new_state = state.copy()
        self.agt.save_experience(old_state, action, reward, new_state)
        if done:
          print("timestamp is %d" % t)
          break

      self.agt.replay()

trainer = Trainer()
trainer.train()
