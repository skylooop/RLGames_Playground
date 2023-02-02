import flappy as flappypack
from absl import app, flags

import numpy as np
import torch
from PIL import Image


FLAGS = flags.FLAGS

def main(_):
    screen, movementInfo = flappypack.main()
    
    for episode in range(10):
        
        flappy = flappypack.FlappyEnv(movementInfo)
        done = None
        reward = 0
        image_state, done, info = flappy.step(False) # like reset
        for t in range(100):
            action = np.random.random() > 0.8
            image_state, done, info = flappy.step(action)
            if done:
                reward = info['score']
                break
            image_state = (image_state[:,:,:3] * [0.2989, 0.5870, 0.1140]).sum(axis=2).transpose(1, 0)
            image_state = Image.fromarray(np.uint8(image_state) , 'L').resize((64, 64))
            image_state = torch.from_numpy(np.asarray(image_state)).unsqueeze(0)
        
        print(f"Episode: {episode}, Reward: {reward}")
            
if __name__ == "__main__":
    app.run(main) 