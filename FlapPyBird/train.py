import flappy as flappypack
from absl import app, flags

import numpy as np
import torch

FLAGS = flags.FLAGS

def main(_):
    screen, movementInfo = flappypack.main()
    
    for episode in range(10):
        flappy = flappypack.FlappyEnv(movementInfo)
        done = None
        reward = 0
        image_state, done, info = flappy.step(False) # like reset
        image_state = torch.from_numpy(image_state)
        for t in range(100):
            #action = np.random.choice(2, size=1,p=np.array([0.7, 0.3]))
            action = np.random.random() > 0.8
            image_state, done, info = flappy.step(action)
            if done:
                reward = info['score']
                break
            
            image_state = torch.from_numpy(image_state)
            print(image_state)
        print(f"Episode: {episode}, Reward: {reward}")
            
if __name__ == "__main__":
    app.run(main) 