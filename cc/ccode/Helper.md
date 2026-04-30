If the distributed function has been put into a helper / util seperate file.
We can simply wrap the following bundle of code as distributed

```Python
if IS_DISTRIBUTED:
    import torch
    if torch.cuda.is_available():
        from imaginaire.utils import distributed
    else:
        import torch.distributed as _torch_dist
        class _gloo_dist:
            def init(self):
                _torch_dist.init_process_group(backend="gloo")
                if _torch_dist.get_rank() == 0:
                    logger.info("No GPU detected, using torch gloo backend for distributed.")
            def get_rank(self): return _torch_dist.get_rank()
            def get_world_size(self): return _torch_dist.get_world_size()
            def gather_object(self, payload, dst=0):
                out = [None] * self.get_world_size() if self.get_rank() == dst else None
                _torch_dist.gather_object(payload, out, dst=dst)
                return out
            def all_gather_object(self, payload):
                out = [None] * self.get_world_size()
                _torch_dist.all_gather_object(out, payload)
                return out
            def broadcast_object(self, obj, src=0):
                obj_list = [obj]
                _torch_dist.broadcast_object_list(obj_list, src=src)
                return obj_list[0]
            def barrier(self): _torch_dist.barrier()
            def destroy_process_group(self): _torch_dist.destroy_process_group()
        distributed = _gloo_dist()
else:
    logger.info("Single rank run, using a dummy distributed class.")
    class dummy_dist:
        def __init__(self): pass
        def init(self): pass
        def get_rank(self): return 0
        def get_world_size(self): return 1
        def gather_object(self, payload, dst=0): return [payload]
        def all_gather_object(self, payload): return [payload]
        def broadcast_object(self, obj, src=0): return obj
        def barrier(self): pass
        def destroy_process_group(self): pass
    distributed = dummy_dist()  # type: ignore[assignment]
```