import xlb, warp as wp, jax, trimesh
print("xlb", getattr(xlb, "__version__", "unknown"))
wp.init()
print("warp devices:", wp.get_devices())
print("jax devices:", jax.devices())
