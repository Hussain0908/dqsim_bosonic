from __future__ import annotations

import pytest

from dqsim import simulate_distributed, simulate_monolithic


class TestMonolithicOptionParsing:
    @pytest.mark.parametrize("mode", ["state_vector", "state-vector", "statevector", "sv", " SV "])
    def test_statevector_mode_aliases_are_accepted_before_circuit_decoding(self, mode: str) -> None:
        with pytest.raises(AttributeError, match="model_dump_json"):
            simulate_monolithic(object(), mode=mode)

    @pytest.mark.parametrize("mode", ["mps", "matrix_product_state", "matrix-product-state", " MPS "])
    def test_mps_mode_aliases_are_accepted_before_circuit_decoding(self, mode: str) -> None:
        with pytest.raises(AttributeError, match="model_dump_json"):
            simulate_monolithic(object(), mode=mode)

    def test_invalid_monolithic_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported monolithic simulation mode"):
            simulate_monolithic(object(), mode="tensor_network")

    def test_statevector_mode_rejects_mps_options(self) -> None:
        with pytest.raises(TypeError, match="not supported"):
            simulate_monolithic(object(), mode="state_vector", max_bond_dimension=1)

    def test_mps_mode_rejects_unknown_option(self) -> None:
        with pytest.raises(TypeError, match="Unsupported MPS option"):
            simulate_monolithic(object(), mode="mps", unknown_option=True)

    def test_mps_mode_accepts_supported_options_before_circuit_decoding(self) -> None:
        with pytest.raises(AttributeError, match="model_dump_json"):
            simulate_monolithic(
                object(),
                mode="mps",
                max_bond_dimension=1,
                truncation_threshold=1e-14,
            )

    def test_mps_mode_accepts_none_bond_dimension_before_circuit_decoding(self) -> None:
        with pytest.raises(AttributeError, match="model_dump_json"):
            simulate_monolithic(object(), mode="mps", max_bond_dimension=None)

    def test_mps_mode_rejects_zero_bond_dimension(self) -> None:
        with pytest.raises(ValueError, match="max_bond_dimension"):
            simulate_monolithic(object(), mode="mps", max_bond_dimension=0)

    @pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
    def test_mps_mode_rejects_invalid_truncation_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="truncation_threshold"):
            simulate_monolithic(object(), mode="mps", truncation_threshold=threshold)


class TestDistributedModeParsing:
    def test_invalid_distributed_mode_raises_before_decoding_circuit(self) -> None:
        with pytest.raises(ValueError, match="Unsupported distributed simulation mode"):
            simulate_distributed(object(), mode="state_vector")
