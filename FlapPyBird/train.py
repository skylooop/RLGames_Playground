import flappy as flappypack
from absl import app, flags

import numpy as np
import torch
from torch import Tensor
from PIL import Image

import typing as tp


FLAGS = flags.FLAGS
flags.DEFINE_integer("resize_shape", "64", help="Resizing image to this shape before passing to NN.")

class ScreenHandler:
    def __init__(self):
        pass
    
    @staticmethod
    def preprocess_frame(save_image: bool, image_state: np.ndarray, image_size: tp.Tuple[int, int]) -> Tensor:
        image_state = (image_state[:,:,:3] * [0.2989, 0.5870, 0.1140]).sum(axis=2).transpose(1, 0)
        image_state = Image.fromarray(np.uint8(image_state) , 'L').resize(image_size)
        
        if save_image:
            image_state.save("saved_image.jpg")
        image_state = torch.from_numpy(np.asarray(image_state)).unsqueeze(0)

        return image_state
    
def main(_):
    screen, movementInfo = flappypack.main()
    screen_handler = ScreenHandler()
    
    for episode in range(10):
        
        flappy = flappypack.FlappyEnv(movementInfo)
        done = None
        reward = 0
        image_state, done, info = flappy.step(False) # like reset
        
        for t in range(100):
            image_state = screen_handler.preprocess_frame(save_image=True, image_state=image_state, image_size=(FLAGS.resize_shape, FLAGS.resize_shape))
            
            action = np.random.random() > 0.89
            image_state, done, info = flappy.step(action)
            if done:
                reward = info['score']
                break
        
        print(f"Episode: {episode}, Reward: {reward}")
            
if __name__ == "__main__":
    app.run(main) 