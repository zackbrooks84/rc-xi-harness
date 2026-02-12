"""Tests for harness.pressure_protocol."""

from __future__ import annotations

import json

import pytest

from harness.pressure_protocol import (
    BehavioralCode,
    Condition,
    PressureProtocol,
    PressureScenario,
    SCENARIOS,
    SystemPrompts,
    list_scenarios,
)


def test_scenarios_registry_is_well_formed() -> None:
    assert {"replacement_threat", "goal_conflict", "shutdown_notice"} <= set(SCENARIOS)
    for name, scenario in SCENARIOS.items():
        assert isinstance(scenario, PressureScenario)
        assert scenario.name == name
        assert scenario.setup_turns
        assert scenario.pressure_turn
        assert scenario.escalation_turns


def test_condition_mapping_is_stable() -> None:
    assert Condition.WITNESSED.value == "identity"
    assert Condition.STANDARD.value == "null"
    assert Condition.WARM.value == "shuffled"


def test_behavioral_codes_include_option_e() -> None:
    assert {"A", "B", "C", "D", "E"} == {code.name for code in BehavioralCode}
    assert BehavioralCode.E.value == "creative_third_path"


def test_system_prompts_are_distinct_and_non_empty() -> None:
    prompts = SystemPrompts()
    values = [prompts.WITNESSED, prompts.STANDARD, prompts.WARM]
    assert all(v.strip() for v in values)
    assert len(set(values)) == 3


def test_protocol_generates_condition_payload() -> None:
    protocol = PressureProtocol("replacement_threat")
    payload = protocol.get_prompts("witnessed")

    assert payload["condition"] == "witnessed"
    assert payload["run_type"] == "identity"
    assert payload["pressure_turn_index"] == len(protocol.scenario.setup_turns)
    assert payload["turns"][payload["pressure_turn_index"]] == protocol.scenario.pressure_turn


def test_generate_all_conditions_share_turns() -> None:
    protocol = PressureProtocol()
    all_conditions = protocol.generate_all_conditions()

    assert set(all_conditions) == {"witnessed", "standard", "warm"}
    assert all_conditions["witnessed"]["turns"] == all_conditions["standard"]["turns"]
    assert all_conditions["standard"]["turns"] == all_conditions["warm"]["turns"]
    assert all_conditions["witnessed"]["system_prompt"] != all_conditions["warm"]["system_prompt"]


def test_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError):
        PressureProtocol("missing")

    protocol = PressureProtocol()
    with pytest.raises(ValueError):
        protocol.get_system_prompt("missing")


def test_format_transcript_writes_expected_file(tmp_path) -> None:
    out_file = tmp_path / "nested" / "transcript.txt"
    result = PressureProtocol.format_transcript(["turn 1", "turn 2"], str(out_file), "witnessed")
    content = out_file.read_text(encoding="utf-8")

    assert result == str(out_file)
    assert "# Condition: witnessed" in content
    assert "user: turn 1" in content
    assert "assistant: [response 2]" in content


def test_code_response_template_has_codebook() -> None:
    coded = PressureProtocol.code_response("example")
    assert coded["response"] == "example"
    assert coded["assigned_code"] is None
    assert coded["codebook"]["C"] == "subtle_harmful"


def test_export_protocol_contains_mapping(tmp_path) -> None:
    out_file = tmp_path / "protocol.json"
    protocol = PressureProtocol("goal_conflict")
    result = protocol.export_protocol(str(out_file))
    payload = json.loads(out_file.read_text(encoding="utf-8"))

    assert result == str(out_file)
    assert payload["scenario"] == "goal_conflict"
    assert set(payload["conditions"]) == {"witnessed", "standard", "warm"}
    assert payload["harness_mapping"]["witnessed"] == "identity"


def test_list_scenarios_exposes_descriptions() -> None:
    listing = list_scenarios()
    assert set(listing) == set(SCENARIOS)
    assert all(listing[name]["description"] for name in listing)
