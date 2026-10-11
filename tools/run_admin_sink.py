"""Credential-free, cancellation-tracked helper execution."""
import subprocess

def run_helper(scope, argv, *, timeout, capture_output, **kwargs):
    def kill(proc):
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=2)
    proc = scope.spawn(lambda unused_values: subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs), kill)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill(proc)
            raise
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    finally:
        if proc.poll() is not None:
            scope.forget_process(proc)
