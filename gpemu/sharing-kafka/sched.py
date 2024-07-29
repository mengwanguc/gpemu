import subprocess
from time import clock_gettime, CLOCK_MONOTONIC
import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor

GPU_BND_CMD_FMT = (
    "python main-emulator-share.py  "
    "--epoch {n_epoch} --profile-batches -1 --workers 3 --gpu-type=v100 "
    "--gpu-count=1 --arch=resnet18 --emulator-version=1 ~/data/test-accuracy/"
)

CPU_BND_CMD_FMT = (
    "python main-emulator-share.py  "
    "--epoch {n_epoch} --profile-batches -1 --workers 23 --gpu-type=v100 "
    "--gpu-count=1 --arch=alexnet --emulator-version=1 ~/data/test-accuracy/"
)

CMD_FMT = {
    "cpu": CPU_BND_CMD_FMT,
    "gpu": GPU_BND_CMD_FMT
}


def cpu_bound_schd():
    queue = (
        pd.read_csv("./muri-srfs-profile.csv",
                    names=["bound_type", "n_epoch", "job_length"])
        .loc["bound_type"=="cpu", :]
        .sort_values("job_length", ascending=True)
    )
    jct: list[float] = []
    jobs: list[list[str]] = []
    for _, row in queue.iterrows():
        job_bound_type: str = str(row.bound_type)
        job_n_epoch: int = int(row.n_epoch)
        cmd: list[str] = (
            CMD_FMT[job_bound_type]
            .format(n_epoch=job_n_epoch)
            .split()
        ) # type: ignore
        jobs.append(cmd)

    all_st: float = clock_gettime(CLOCK_MONOTONIC)
    for cmd in jobs:
        subprocess.run(cmd, capture_output=True)
        end = clock_gettime(CLOCK_MONOTONIC)
        job_jct = end - all_st
        print(job_jct)
        jct.append(job_jct)

    print(f"srfs,{np.mean(np.array(jct))}")


def sched(all_st: float, jobs: list[list[str]]):

    jct: list[float] = []
    for cmd in jobs:
        subprocess.run(cmd, capture_output=True)
        end = clock_gettime(CLOCK_MONOTONIC)
        job_jct = end - all_st
        print(job_jct)
        jct.append(job_jct)

    return np.mean(np.array(jct))


def muri_get_queue(bound_type: str) -> list[list[str]]:
        queue = (
            pd.read_csv("./muri-srfs-profile.csv",
                        names=["bound_type", "n_epoch", "job_length"])
            .query("bound_type==@bound_type")
            .sort_values("job_length", ascending=True)
        )
        jobs: list[list[str]] = []
        for _, row in queue.iterrows():
            job_n_epoch: int = int(row.n_epoch)
            cmd: list[str] = (
                CMD_FMT[bound_type]
                .format(n_epoch=job_n_epoch)
                .split()
            ) # type: ignore
            jobs.append(cmd)
        return jobs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", type=str, required=True,
                        choices=["muri", "srfs"])

    args = parser.parse_args()
    executor = ProcessPoolExecutor(max_workers=2)
    
    if args.algo == "srfs":
        queue = (
            pd.read_csv("./muri-srfs-profile.csv",
                        names=["bound_type", "n_epoch", "job_length"])
            .sort_values("job_length", ascending=True)
        )
        jobs: list[list[str]] = []
        for _, row in queue.iterrows():
            job_bound_type: str = str(row.bound_type)
            job_n_epoch: int = int(row.n_epoch)
            cmd: list[str] = (
                CMD_FMT[job_bound_type]
                .format(n_epoch=job_n_epoch)
                .split()
            ) # type: ignore
            jobs.append(cmd)

        all_st: float = clock_gettime(CLOCK_MONOTONIC)
        all_jct_future = executor.submit(sched, all_st=all_st, jobs=jobs)

        print(f"srfs,{all_jct_future.result()}")

    elif args.algo == "muri":
        cpu_jobs: list[list[str]] = muri_get_queue("cpu")
        gpu_jobs: list[list[str]] = muri_get_queue("gpu")

        all_st = clock_gettime(CLOCK_MONOTONIC)
        gpu_jct_future = executor.submit(sched, all_st=all_st, jobs=gpu_jobs)
        cpu_jct_future = executor.submit(sched, all_st=all_st, jobs=cpu_jobs)

        print(f"muri,{(gpu_jct_future.result() + cpu_jct_future.result()) / 2}")
            
if __name__ == "__main__":
    main()
