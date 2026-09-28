import subprocess

import pytest

from transformersci import runners

CPUINFO = "processor\t: 0\nmodel name\t: Intel(R) Xeon(R)  Platinum 8488C\n\n"
MEMINFO = "MemTotal:       65011712 kB\nMemFree:  1 kB\n"


def fake_read(files: dict[str, str]):
    def read(path: str) -> str:
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    return read


@pytest.mark.parametrize(
    ("name", "group", "expected"),
    [
        # Real names from huggingface/transformers job logs (2026-09-27/28).
        (
            "aws-g5-4xlarge-cache-use1-public-80-x9tqf-runner-7bthr",
            "",
            "aws-g5-4xlarge-cache",
        ),
        ("aws-m8i-l-cache-use1-public-80-78nkj-runner-twjx2", "", "aws-m8i-l-cache"),
        (
            "hfc-amd-mi355-2gpu-use1-hybrid-80-tv4nc-runner-c4kmd",
            "",
            "hfc-amd-mi355-2gpu",
        ),
        # GitHub's group wins when there is one.
        ("whatever", "aws-general-8-plus", "aws-general-8-plus"),
        ("GitHub Actions 12", "GitHub Actions", runners.GITHUB_HOSTED),
        # A queued job has no runner yet; an unknown name shape is not guessed.
        ("", "", ""),
        ("my-laptop", "", ""),
    ],
)
def test_runner_type(name: str, group: str, expected: str) -> None:
    assert runners.runner_type(name, group) == expected


def test_probe_reads_the_container_view_and_nvidia_smi() -> None:
    read = fake_read(
        {
            "/proc/cpuinfo": CPUINFO,
            "/proc/meminfo": MEMINFO,
            # A pod limited to 7.5 CPUs and 30 GiB on a bigger machine.
            "/sys/fs/cgroup/cpu.max": "750000 100000\n",
            "/sys/fs/cgroup/memory.max": f"{30 * 2**30}\n",
        }
    )

    def run(command: list[str]) -> str:
        assert command[0] == "nvidia-smi"
        return "NVIDIA A10G, 23028\nNVIDIA A10G, 23028\n"

    assert runners.probe_hardware(read=read, run=run) == {
        "cpu_model": "Intel(R) Xeon(R) Platinum 8488C",
        "vcpus": "7.5",
        "memory_gib": "30",
        "gpu_vendor": "nvidia",
        "gpu_model": "NVIDIA A10G",
        "gpu_count": "2",
        "gpu_memory_gib": "22",
    }


def test_probe_falls_back_to_the_machine_and_skips_missing_facts() -> None:
    read = fake_read(
        {
            "/proc/meminfo": MEMINFO,
            "/sys/fs/cgroup/cpu.max": "max 100000\n",
            "/sys/fs/cgroup/memory.max": "max\n",
        }
    )

    def run(command: list[str]) -> str:
        raise FileNotFoundError("nvidia-smi")

    facts = runners.probe_hardware(read=read, run=run)
    # No /proc/cpuinfo, no GPU: those facts are absent, not empty strings.
    assert "cpu_model" not in facts and "gpu_count" not in facts
    assert facts["memory_gib"] == "62"
    assert float(facts["vcpus"]) >= 1


def test_probe_survives_a_hung_nvidia_smi() -> None:
    def run(command: list[str]) -> str:
        raise subprocess.TimeoutExpired(command, 5)

    assert "gpu_count" not in runners.probe_hardware(read=fake_read({}), run=run)


def test_resource_attributes_only_on_a_github_runner() -> None:
    probe = lambda: {"cpu_model": "Xeon, v4 = fast", "vcpus": "8"}  # noqa: E731
    assert runners.runner_resource_attributes({}, probe=probe) == []
    attributes = runners.runner_resource_attributes(
        {"RUNNER_NAME": "aws-m8i-l-cache-use1-public-80-78nkj-runner-twjx2"},
        probe=probe,
    )
    assert attributes == [
        "transformers.test.runner.name=aws-m8i-l-cache-use1-public-80-78nkj-runner-twjx2",
        "transformers.test.runner.type=aws-m8i-l-cache",
        # Commas and "=" would break OTEL_RESOURCE_ATTRIBUTES.
        "transformers.test.runner.cpu_model=Xeon  v4   fast",
        "transformers.test.runner.vcpus=8",
    ]


def test_resource_attributes_never_raise() -> None:
    def probe() -> dict[str, str]:
        raise RuntimeError("boom")

    assert runners.runner_resource_attributes({"RUNNER_NAME": "x"}, probe=probe) == []
