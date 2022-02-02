from subprocess import Popen, PIPE
import time
import sys

def launch(cmd, num=1, env=None):
    running_procs = [(i, Popen(cmd(i), stdout=PIPE, stderr=PIPE, env=env(i) if env else None))
                     for i in range(num)]

    while running_procs:
        for idx, proc in running_procs:
            retcode = proc.poll()
            if retcode is not None:  # Process finished.
                running_procs.remove((idx, proc))
                if retcode != 0:
                    """Error handling."""
                    print(f"{idx} err: {proc.stderr.read()}")
                print(f"{idx} finish: {proc.stdout.read()}")
                break
            else:  # No process is done, wait a bit and check again.
                print(f"{idx} running: {proc.stdout.read()}")
                print('.', end='')
                time.sleep(.1)
                continue


if __name__ == '__main__':
    launch(lambda s: sys.argv[2:], int(sys.argv[1]))