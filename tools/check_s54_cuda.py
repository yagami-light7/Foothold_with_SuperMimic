"""Check actual CUDA execution before creating a training job."""
import json
import sys


def main():
    try:
        import torch
        x = torch.ones(1, device='cuda:0')
        value = (x + x).item()
        torch.cuda.synchronize()
        if value != 2.:
            raise RuntimeError(f'Unexpected CUDA computation result: {value}')
        print(json.dumps(dict(status='PASS', gpu=torch.cuda.get_device_name(0),
                              torch=torch.__version__, cuda=torch.version.cuda)), flush=True)
    except Exception as exc:
        print(f'CUDA preflight FAILED: {type(exc).__name__}: {exc}\n'
              'Training was not started. nvidia-smi alone does not verify CUDA execution.\n'
              'After a suspend-related failure, restart the training container and retry this check; '
              'if it still fails, reboot the host and check again before resuming.', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
