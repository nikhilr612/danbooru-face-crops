import sys, numpy as np
sys.path.insert(0, ".")
import afaces

root = sys.argv[1]
ds = afaces.load(root, "val", fields=("image", "tags", "post_id", "flags"))
assert len(ds) == len(np.load(f"{root}/val/post_id.npy", mmap_mode="r"))
r = ds[3]
assert r["image"].shape == (128, 128, 3) and r["image"].dtype == np.uint8 and r["tags"].dtype == np.int8
b = ds.batch(np.array([5, 1, 5, 0]))
assert (b["post_id"] == [ds[5]["post_id"], ds[1]["post_id"], ds[5]["post_id"], ds[0]["post_id"]]).all()
assert len(ds.subset(100)) == 100 and len(ds[2:6]["image"]) == 4
assert len(afaces.reliable_tags(root)) == 43
print("ok", len(ds), afaces.unpack_flags(ds[:1000]["flags"])["frame_shifted"].mean())
