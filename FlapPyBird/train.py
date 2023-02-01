import flappy
from absl import app, flags


FLAGS = flags.FLAGS

def main(_):
    flappy.main()


if __name__ == "__main__":
    app.run(main)