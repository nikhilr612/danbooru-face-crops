"""The same skeleton with JAX and Grain. Install a GPU build of JAX yourself (`pip install "jax[cuda13]"`); the `jax` extra installs the CPU one.

The "model" is a placeholder that returns real validation images, so the headline should stay near 0. Select checkpoints on the "val" reference; report the final model once on "test".
"""
import dbanifaces
from dbanifaces.jax import Evaluator, loader
from dbanifaces.metrics import Reference


def main():
    train = dbanifaces.load("train", fields=("image", "tags"))
    val = dbanifaces.load("val", fields=("image",))
    evaluator = Evaluator(Reference.load(split="val"))

    for step, batch in enumerate(loader(train, batch_size=128, seed=0, num_epochs=None, worker_count=4)):
        images = batch["image"]  # numpy uint8 [B, 128, 128, 3]; batch["tags"] is int8 [B, 65]
        # state = train_step(state, images, batch["tags"]); ...
        if step % 1000 == 0:
            evaluator.reset()
            evaluator.update(val[:2000]["image"])  # placeholder for generated samples; the same N every time
            print(step, {k: round(v, 3) for k, v in evaluator.compute().items()})
        if step >= 2000:
            break


if __name__ == "__main__":  # worker processes re-import this file
    main()
