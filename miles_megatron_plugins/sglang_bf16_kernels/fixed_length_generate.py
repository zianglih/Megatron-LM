"""Force a fixed response length so the scoring prefill is exactly 8192 tokens."""

from __future__ import annotations

from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate


async def generate_fixed_length(input: GenerateFnInput) -> GenerateFnOutput:
    params = deepcopy(input.sampling_params)
    params["ignore_eos"] = True
    params["max_new_tokens"] = input.args.rollout_max_response_len
    output = await generate(
        GenerateFnInput(
            state=input.state,
            sample=input.sample,
            sampling_params=params,
            evaluation=input.evaluation,
        )
    )
    sample = output.samples
    if isinstance(sample, list):
        raise RuntimeError("Fixed-length parity generation expects exactly one sample")
    if sample.response_length != input.args.rollout_max_response_len:
        raise RuntimeError(
            "Fixed-length parity rollout returned "
            f"{sample.response_length} tokens, expected {input.args.rollout_max_response_len}"
        )
    return output


__all__ = ["generate_fixed_length"]
