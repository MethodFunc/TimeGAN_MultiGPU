import yaml
from argparse import Namespace
from load_data import dataloader

from exp.exp_main import Experience



with open("setting.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = Namespace(**config)


if __name__ == "__main__":
    raw_data = dataloader(config)

    config.seq_length = raw_data.shape[-1]


    exp = Experience(raw_data,  config)
    print("model train start")
    exp.train_model()

    print("model test start")
    exp.test_model()

    exp.save_model("wind_generator_10s")