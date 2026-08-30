from freetoken.utils.torch_utils import nvtx_annotate


def test_nvtx_annotation_is_optional_on_cpu():
    class Operation:
        @nvtx_annotate("cpu-reference")
        def run(self, value: int) -> int:
            return value + 1

    assert Operation().run(41) == 42
