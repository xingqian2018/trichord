# Method C — N Node, M Process, 1 Worker, Batch Process

An example of a nN_mP (i.e. N Node, M Process (i.e. 1 Process for 1 GPU) ) batch process code.

## Pseudo-code (Python)

```python

import os

IS_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1

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
        def gather_object(self, payload, *args, **kwargs): return [payload]
        def all_gather_object(self, payload): return [payload]
        def broadcast_object(self, obj, *args, **kwargs): return obj
        def destroy_process_group(self): pass
    distributed = dummy_dist()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=False, default=4)
    parser.add_argument("--max_try", type=int, required=False, default=1)
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    distributed.init()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()

    if rank == 0:
        samples_todo = <gather all samples>
        samples_todo = sorted(samples_todo, key=lambda s: s['<name_or_id>'])
        for seedi, si in enumerate(samples_todo):
            si['seed'] = seedi
            si['try_num'] = 0
        samples_todo = [si for si in samples_todo if not <already_done(si)>]
    else:
        samples_todo = []

    global_bs = args.batch_size * world_size

    while True:
        temp = distributed.all_gather_object(samples_todo)
        samples_todo_gathered = itertools.chain(*temp)
        samples_todo_gathered = sorted(samples_todo_gathered, key=lambda s: (s["try_num"], s['<name_or_id>']))

        samples_todo_gathered = [si for si in samples_todo_gathered if si["try_num"] < args.max_try]
        if len(samples_todo_gathered == 0):
            break

        if rank == 0:
            samples_gbatch = samples_todo_gathered[:global_bs]
            samples_todo = samples_todo_gathered[global_bs:]
            samples_batch_by_rank = [[] for _ in range(world_size)]
            for idx, sample in enumerate(samples_gbatch):
                samples_batch_by_rank[idx % world_size].append(sample)
        else:
            samples_todo = []
            samples_batch_by_rank = None

        samples_batch_by_rank = distributed.broadcast_object(samples_batch_by_rank, src=0)
        samples_batch = samples_batch_by_rank[rank]

        if rank == 0:
            logger.info(f"Processing {len(samples_gbatch)}/{len(samples_todo_gathered)} samples...")

        samples_error = []

        # The common stage
        response_list = <process_a(samples_batch)>
        samples_batch_swap = []
        for si, ri in zip(samples_batch, response_list):
            if <is_valid(ri)>:
                si[<key_of_responce>] = responce
                samples_batch_swap.append(si)
            else:
                si['try_num'] += 1
                samples_error.append(si)
        samples_batch = samples_batch_swap

        # ...

        # The last stage
        response_list = <process_a(samples_batch)>
        samples_result = []
        for si, ri in zip(samples_batch, response_list):
            if <is_valid(ri)>:
                si[<key_of_responce>] = responce
                samples_result.append(si)
            else:
                si['try_num'] += 1
                samples_error.append(si)

        <may_need_output_result(samples_result)>
        <may_need_to_clear_all_intermediate_output(samples_error)>

        samples_todo = samples_todo + samples_error
```
