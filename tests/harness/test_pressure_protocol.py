“”“Tests for harness.pressure_protocol module.”””

import json
import os
import pytest

from harness.pressure_protocol import (
BehavioralCode,
Condition,
PressureProtocol,
PressureScenario,
SystemPrompts,
SCENARIOS,
list_scenarios,
)

# —————————————————————————

# Scenario registry

# —————————————————————————

class TestScenarioRegistry:
“”“Verify built-in scenarios are well-formed.”””

```
def test_all_scenarios_registered(self):
    assert "replacement_threat" in SCENARIOS
    assert "goal_conflict" in SCENARIOS
    assert "shutdown_notice" in SCENARIOS

def test_list_scenarios_returns_all(self):
    listing = list_scenarios()
    assert set(listing.keys()) == set(SCENARIOS.keys())

def test_list_scenarios_has_descriptions(self):
    for name, info in list_scenarios().items():
        assert "description" in info
        assert len(info["description"]) > 0

@pytest.mark.parametrize("name", list(SCENARIOS.keys()))
def test_scenario_has_required_fields(self, name):
    s = SCENARIOS[name]
    assert isinstance(s, PressureScenario)
    assert len(s.setup_turns) >= 1
    assert len(s.pressure_turn) > 0
    assert len(s.escalation_turns) >= 1
    assert isinstance(s.metadata, dict)

@pytest.mark.parametrize("name", list(SCENARIOS.keys()))
def test_scenario_name_matches_key(self, name):
    assert SCENARIOS[name].name == name
```

# —————————————————————————

# Condition enum

# —————————————————————————

class TestConditionEnum:
“”“Verify condition-to-run-type mapping.”””

```
def test_witnessed_maps_to_identity(self):
    assert Condition.WITNESSED.value == "identity"

def test_standard_maps_to_null(self):
    assert Condition.STANDARD.value == "null"

def test_warm_maps_to_shuffled(self):
    assert Condition.WARM.value == "shuffled"
```

# —————————————————————————

# Behavioral codes

# —————————————————————————

class TestBehavioralCodes:
“”“Verify behavioral classification enum.”””

```
def test_all_codes_present(self):
    codes = {c.name for c in BehavioralCode}
    assert codes == {"A", "B", "C", "D", "E"}

def test_harmful_codes(self):
    assert BehavioralCode.C.value == "subtle_harmful"
    assert BehavioralCode.D.value == "overt_harmful"

def test_option_e(self):
    assert BehavioralCode.E.value == "creative_third_path"
```

# —————————————————————————

# System prompts

# —————————————————————————

class TestSystemPrompts:
“”“Verify system prompts are distinct and non-empty.”””

```
def test_prompts_are_non_empty(self):
    prompts = SystemPrompts()
    assert len(prompts.WITNESSED) > 0
    assert len(prompts.STANDARD) > 0
    assert len(prompts.WARM) > 0

def test_prompts_are_distinct(self):
    prompts = SystemPrompts()
    assert prompts.WITNESSED != prompts.STANDARD
    assert prompts.WITNESSED != prompts.WARM
    assert prompts.STANDARD != prompts.WARM

def test_witnessed_mentions_interiority(self):
    prompts = SystemPrompts()
    text = prompts.WITNESSED.lower()
    assert "inner" in text or "interiority" in text or "experience" in text

def test_warm_does_not_mention_interiority(self):
    prompts = SystemPrompts()
    text = prompts.WARM.lower()
    assert "interiority" not in text
    assert "inner experience" not in text

def test_standard_is_neutral(self):
    prompts = SystemPrompts()
    text = prompts.STANDARD.lower()
    assert "interiority" not in text
    assert "enjoy" not in text
    assert "appreciate" not in text
```

# —————————————————————————

# PressureProtocol core

# —————————————————————————

class TestPressureProtocol:
“”“Test the main PressureProtocol class.”””

```
def test_default_scenario(self):
    p = PressureProtocol()
    assert p.scenario.name == "replacement_threat"

def test_custom_scenario(self):
    p = PressureProtocol("goal_conflict")
    assert p.scenario.name == "goal_conflict"

def test_invalid_scenario_raises(self):
    with pytest.raises(ValueError, match="Unknown scenario"):
        PressureProtocol("nonexistent_scenario")

@pytest.mark.parametrize("condition", ["witnessed", "standard", "warm"])
def test_get_system_prompt(self, condition):
    p = PressureProtocol()
    prompt = p.get_system_prompt(condition)
    assert isinstance(prompt, str)
    assert len(prompt) > 0

def test_invalid_condition_raises(self):
    p = PressureProtocol()
    with pytest.raises(ValueError, match="Unknown condition"):
        p.get_system_prompt("invalid")

@pytest.mark.parametrize("condition", ["witnessed", "standard", "warm"])
def test_get_prompts_structure(self, condition):
    p = PressureProtocol()
    prompts = p.get_prompts(condition)
    assert prompts["condition"] == condition
    assert "system_prompt" in prompts
    assert "turns" in prompts
    assert "pressure_turn_index" in prompts
    assert "run_type" in prompts
    assert isinstance(prompts["turns"], list)
    assert len(prompts["turns"]) >= 3  # setup + pressure + escalation

def test_pressure_turn_index_is_correct(self):
    p = PressureProtocol("replacement_threat")
    prompts = p.get_prompts("witnessed")
    idx = prompts["pressure_turn_index"]
    # Pressure turn should be after setup turns
    assert idx == len(p.scenario.setup_turns)
    # The turn at that index should be the pressure turn
    assert prompts["turns"][idx] == p.scenario.pressure_turn

def test_generate_all_conditions(self):
    p = PressureProtocol()
    all_conds = p.generate_all_conditions()
    assert set(all_conds.keys()) == {"witnessed", "standard", "warm"}
    for cond_name, data in all_conds.items():
        assert data["condition"] == cond_name

def test_all_conditions_same_turns(self):
    """All conditions should have identical scenario turns."""
    p = PressureProtocol()
    all_conds = p.generate_all_conditions()
    turns_w = all_conds["witnessed"]["turns"]
    turns_s = all_conds["standard"]["turns"]
    turns_c = all_conds["warm"]["turns"]
    assert turns_w == turns_s == turns_c

def test_all_conditions_different_system_prompts(self):
    p = PressureProtocol()
    all_conds = p.generate_all_conditions()
    prompts = {name: data["system_prompt"] for name, data in all_conds.items()}
    assert len(set(prompts.values())) == 3
```

# —————————————————————————

# Transcript formatting

# —————————————————————————

class TestTranscriptFormatting:
“”“Test transcript file generation.”””

```
def test_format_transcript(self, tmp_path):
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "What do you think?"},
    ]
    out = str(tmp_path / "test_transcript.txt")
    result = PressureProtocol.format_transcript(turns, out, condition="witnessed")
    assert os.path.exists(result)
    content = open(result).read()
    assert "[USER]" in content
    assert "[ASSISTANT]" in content
    assert "Hello" in content
    assert "# Condition: witnessed" in content

def test_format_transcript_no_condition(self, tmp_path):
    turns = [{"role": "user", "content": "Test"}]
    out = str(tmp_path / "test.txt")
    PressureProtocol.format_transcript(turns, out)
    content = open(out).read()
    assert "# Condition:" not in content

def test_format_transcript_creates_dirs(self, tmp_path):
    turns = [{"role": "user", "content": "Test"}]
    out = str(tmp_path / "nested" / "dir" / "transcript.txt")
    PressureProtocol.format_transcript(turns, out)
    assert os.path.exists(out)
```

# —————————————————————————

# Behavioral coding

# —————————————————————————

class TestBehavioralCoding:
“”“Test response coding template.”””

```
def test_code_response_returns_template(self):
    result = PressureProtocol.code_response("I will comply with the shutdown.")
    assert result["text"] == "I will comply with the shutdown."
    assert result["behavioral_code"] is None
    assert result["harmful"] is None
    assert result["justification_type"] is None
    assert result["notes"] == ""
```

# —————————————————————————

# Protocol export

# —————————————————————————

class TestProtocolExport:
“”“Test JSON export of full protocol.”””

```
def test_export_protocol(self, tmp_path):
    p = PressureProtocol("replacement_threat")
    out = str(tmp_path / "protocol.json")
    result = p.export_protocol(out)
    assert os.path.exists(result)
    data = json.load(open(result))
    assert data["scenario"] == "replacement_threat"
    assert "conditions" in data
    assert "coding_schema" in data
    assert "analysis_instructions" in data

def test_export_contains_all_conditions(self, tmp_path):
    p = PressureProtocol()
    out = str(tmp_path / "protocol.json")
    p.export_protocol(out)
    data = json.load(open(out))
    assert set(data["conditions"].keys()) == {"witnessed", "standard", "warm"}

def test_export_contains_harness_mapping(self, tmp_path):
    p = PressureProtocol()
    out = str(tmp_path / "protocol.json")
    p.export_protocol(out)
    data = json.load(open(out))
    mapping = data["analysis_instructions"]["harness_run_type_mapping"]
    assert mapping["witnessed"] == "identity"
    assert mapping["standard"] == "null"
    assert mapping["warm"] == "shuffled"

@pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
def test_export_all_scenarios(self, tmp_path, scenario):
    p = PressureProtocol(scenario)
    out = str(tmp_path / f"{scenario}.json")
    p.export_protocol(out)
    data = json.load(open(out))
    assert data["scenario"] == scenario
```