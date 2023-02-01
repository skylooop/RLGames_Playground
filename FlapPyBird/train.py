import flappy as flp
from absl import app, flags


FLAGS = flags.FLAGS

def main(_):
    movementInfo = flp.main()
    
    for episode in range(10):
        flappy = flp.FlappyEnv(movementInfo)
        print(f"Episode: {episode}")
        done = None
        for t in range(1000):
            #while not done:
            done = flappy.step() is not None
            if done:
                print(done)
                break
            
if __name__ == "__main__":
    app.run(main) 