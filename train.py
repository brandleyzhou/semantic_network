from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import SegOptions

options = SegOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
