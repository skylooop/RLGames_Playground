import flappy as flappypack
from absl import app, flags

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from collections import deque

import typing as tp


FLAGS = flags.FLAGS
flags.DEFINE_integer("resize_shape", "64", help="Resizing image to this shape before passing to NN.")
flags.DEFINE_integer("num_episodes", "10", help="Number of episodes.")


class ScreenHandler:
    def __init__(self, stacked_frames: int):
        self.stacked_frames = stacked_frames
        self.deque = deque()
        
        for i in range(self.stacked_frames):
            self.deque.append(torch.zeros(FLAGS.resize_shape, FLAGS.resize_shape))
            
    def stacker(self, image_state: Tensor) -> Tensor:
        self.deque.append(image_state)
        if len(self.deque) > self.stacked_frames:
            self.deque.popleft()
        frame_stack_stacked = torch.stack(list(self.deque), dim=0)
        
        return frame_stack_stacked
        
            
    
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
    screen_handler = ScreenHandler(stacked_frames = 4)

    for episode in range(FLAGS.num_episodes):
        
        flappy = flappypack.FlappyEnv(movementInfo)
        done, reward = None, 0.0
        image_state, done, info = flappy.step(False) # like reset
        
        for t in range(100):            
            action = np.random.random() > 0.89
            image_state, done, info = flappy.step(action)
            image_state = screen_handler.preprocess_frame(save_image=True, image_state=image_state, image_size=(FLAGS.resize_shape, FLAGS.resize_shape)).squeeze(0)
            image_state_stacked = screen_handler.stacker(image_state=image_state)
            
            print(image_state_stacked.shape)
            if done:
                reward = info['score']
                break
        
        print(f"Episode: {episode}, Reward: {reward}")
            
if __name__ == "__main__":
    app.run(main) 