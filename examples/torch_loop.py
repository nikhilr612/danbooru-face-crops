"""A training loop skeleton: DataLoader batches, and the Evaluator on the validation reference every few steps.

The model here is a placeholder that "generates" real validation images, so the headline should stay near 0. Replace `sample` with your generator (uint8 [B, 128, 128, 3] or float in [-1, 1]).
Select checkpoints on the "val" reference; report the final model once on "test".
"""

import dbanifaces
from dbanifaces.metrics import Reference
from dbanifaces.torch import Evaluator, dataloader


def main():
    train = dbanifaces.load("train", fields=("image", "tags"))
    loader = dataloader(train, batch_size=128, shuffle=True, num_workers=4, seed=0)
    val = dbanifaces.load("val", fields=("image",))

    evaluator = Evaluator(Reference.load(split="val"))  # downloads the weights and the statistics once


    def sample(n: int):
        return val[:n]["image"]  # placeholder for model.generate(n)


    for step, batch in enumerate(loader):
        images = batch["image"]  # uint8 [B, 128, 128, 3]; batch["tags"] is int8 [B, 65]
        x = images.float().div(127.5).sub(1).permute(0, 3, 1, 2)  # NCHW in [-1, 1] for the model
        # loss = model(x, batch["tags"]); ...
        if step % 1000 == 0:
            evaluator.reset()
            evaluator.update(sample(2000))  # use the same N every time you compare runs
            print(step, {k: round(v, 3) for k, v in evaluator.compute().items()})
        if step >= 2000:
            break


if __name__ == "__main__":  # worker processes re-import this file
    main()
