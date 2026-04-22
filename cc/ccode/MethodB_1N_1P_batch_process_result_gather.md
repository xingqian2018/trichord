# Method B — 1 Node, 1 Process, Batch Process, Result Gather

An example of a 1N_1P (i.e. 1 Node, 1 Process) batch process code and result gathering as final output..

## Pseudo-code (Python)

```python

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=False, default=4)
    parser.add_argument("--max_try", type=int, required=False, default=1)
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()

    samples_todo = <gather all samples>
    samples_todo = sorted(samples_todo, key=lambda s: s['<name_or_id>'])
    for seedi, si in enumerate(samples_todo):
        si['seed'] = seedi
        si['try_num'] = 0
    samples_todo = [si for si in samples_todo if not <already_done(si)>]

    result_storage = <list or dict depand on need>
    while True:
        samples_todo = sorted(samples_todo, key=lambda s: (s["try_num"], s['<name_or_id>']))
        samples_todo = [si for si in samples_todo if si["try_num"] < args.max_try]
        if len(samples_todo == 0):
            break

        samples_batch = samples_todo[:batch_size]
        samples_todo = samples_todo[batch_size:]
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
        for si, ri in zip(samples_batch, response_list):
            if <is_valid(ri)>:
                si[<key_of_responce>] = responce
                all_valid_info = <gather_info(si)>
                <store_result(all_valid_info -> result_storage)>
            else:
                si['try_num'] += 1
                samples_error.append(si)

        <may_need_to_clear_all_intermediate_output(samples_error)>

        samples_todo = samples_todo + samples_error

    <process_and_save(result_storage)>
```
