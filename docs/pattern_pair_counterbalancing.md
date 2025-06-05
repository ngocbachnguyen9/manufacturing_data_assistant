## docs/pattern_pair_counterbalancing.md

````markdown
# Pattern-Pair Counterbalancing Design

## Overview

This document details the innovative 3×3 pattern-pair counterbalancing design used in the human study component of the manufacturing data assistant research. This approach provides superior experimental control compared to traditional Latin square designs.

## Design Innovation

### Traditional vs. Pattern-Pair Approach

**Traditional Latin Square Limitations**:
- Simple rotation of conditions across participants
- Limited control over individual exposure distribution
- Potential for uneven condition representation
- Difficulty ensuring balanced global counts

**Pattern-Pair Advantages**:
- Systematic control of both prompt complexity and data quality distributions
- Guaranteed balanced exposure across all participants
- Perfect global count achievement
- Enhanced statistical power through controlled variance

## Mathematical Foundation

### Design Parameters

- **Participants (P)**: 9
- **Tasks per Participant (K)**: 10
- **Total Tasks**: 90
- **Prompt Complexity Levels**: 3 (Easy, Medium, Hard)
- **Data Quality Conditions**: 4 (Q0, Q1, Q2, Q3)
- **Total Conditions**: 12 (3×4 factorial design)

### Target Distribution Goals

**Global Targets**:
- Easy (E): 30 tasks total
- Medium (M): 30 tasks total  
- Hard (H): 30 tasks total
- Baseline (Q0): 45 tasks total
- Error conditions (Q1, Q2, Q3): 15 tasks each

## Pattern Definition

### Quality Patterns (PQ1-PQ3)

**PQ1 - Baseline Heavy**:
Q0: 5 tasks (50%)
Q1: 2 tasks (20%)
Q2: 2 tasks (20%)
Q3: 1 task  (10%)

**PQ2 - Balanced Distribution**:
Q0: 5 tasks (50%)
Q1: 2 tasks (20%)
Q2: 1 task  (10%)
Q3: 2 tasks (20%)

**PQ3 - Error Heavy**:
Q0: 5 tasks (50%)
Q1: 1 task  (10%)
Q2: 2 tasks (20%)
Q3: 2 tasks (20%)

### Prompt Patterns (PC1-PC3)

**PC1 - Easy Heavy**:
E: 4 tasks (40%)
M: 3 tasks (30%)
H: 3 tasks (30%)

**PC2 - Medium Heavy**:
E: 3 tasks (30%)
M: 4 tasks (40%)
H: 3 tasks (30%)

**PC3 - Hard Heavy**:
E: 3 tasks (30%)
M: 3 tasks (30%)
H: 4 tasks (40%)

## Participant Assignment Matrix

### 3×3 Counterbalancing Grid

Quality***REMOVED***Prompt    PC1 (E4,M3,H3)   PC2 (E3,M4,H3)   PC3 (E3,M3,H4)
─────────────────────────────────────────────────────────────────
PQ1 (Q0×5,Q1×2,     P1              P2              P3
Q2×2,Q3×1)

PQ2 (Q0×5,Q1×2,     P4              P5              P6
Q2×1,Q3×2)

PQ3 (Q0×5,Q1×1,     P7              P8              P9
Q2×2,Q3×2)

### Individual Participant Profiles

**Participant P1 (PQ1+PC1)**:
- Quality List: [Q0,Q0,Q0,Q0,Q0,Q1,Q1,Q2,Q2,Q3]
- Prompt List: [E,E,E,E,M,M,M,H,H,H]
- Pairing: Random shuffle of both lists, then zip together

**Participant P5 (PQ2+PC2)**:
- Quality List: [Q0,Q0,Q0,Q0,Q0,Q1,Q1,Q2,Q3,Q3]
- Prompt List: [E,E,E,M,M,M,M,H,H,H]
- Pairing: Random shuffle of both lists, then zip together

**[Similar patterns for P2-P4, P6-P9]**

## Implementation Algorithm

### Task Assignment Process

```python
def generate_participant_tasks(participant_id, quality_pattern, prompt_pattern):
    """Generate 10 tasks for a participant using pattern-pair design"""
    
    # Step 1: Create quality and prompt lists based on patterns
    quality_list = create_quality_list(quality_pattern)
    prompt_list = create_prompt_list(prompt_pattern)
    
    # Step 2: Shuffle each list independently
    random.seed(participant_id + global_seed)
    random.shuffle(quality_list)
    random.shuffle(prompt_list)
    
    # Step 3: Pair quality and prompt conditions
    paired_tasks = list(zip(prompt_list, quality_list))
    
    # Step 4: Final randomization of task order
    random.shuffle(paired_tasks)
    
    return paired_tasks

def create_quality_list(pattern):
    """Convert quality pattern to task list"""
    if pattern == "PQ1":
        return ["Q0"]*5 + ["Q1"]*2 + ["Q2"]*2 + ["Q3"]*1
    elif pattern == "PQ2":
        return ["Q0"]*5 + ["Q1"]*2 + ["Q2"]*1 + ["Q3"]*2
    elif pattern == "PQ3":
        return ["Q0"]*5 + ["Q1"]*1 + ["Q2"]*2 + ["Q3"]*2

def create_prompt_list(pattern):
    """Convert prompt pattern to task list"""
    if pattern == "PC1":
        return ["E"]*4 + ["M"]*3 + ["H"]*3
    elif pattern == "PC2":
        return ["E"]*3 + ["M"]*4 + ["H"]*3
    elif pattern == "PC3":
        return ["E"]*3 + ["M"]*3 + ["H"]*4


Validation Framework - Global Count Verification

Prompt Complexity Totals:

Easy (E):   (4×3) + (3×3) + (3×3) = 12 + 9 + 9 = 30 ✓
Medium (M): (3×3) + (4×3) + (3×3) = 9 + 12 + 9 = 30 ✓  
Hard (H):   (3×3) + (3×3) + (4×3) = 9 + 9 + 12 = 30 ✓

Q0: 5 × 9 participants = 45 ✓
Q1: (2×3) + (2×3) + (1×3) = 6 + 6 + 3 = 15 ✓
Q2: (2×3) + (1×3) + (2×3) = 6 + 3 + 6 = 15 ✓
Q3: (1×3) + (2×3) + (2×3) = 3 + 6 + 6 = 15 ✓

Implementation Validation

Pre-Study Validation Checks

def validate_counterbalancing(assignments):
    """Comprehensive validation of pattern-pair design"""
    
    # Check 1: Total task count
    total_tasks = sum(len(tasks) for tasks in assignments.values())
    assert total_tasks == 90, f"Expected 90 tasks, got {total_tasks}"
    
    # Check 2: Global prompt complexity distribution
    prompt_counts = count_conditions(assignments, 'prompt')
    assert prompt_counts == {'E': 30, 'M': 30, 'H': 30}
    
    # Check 3: Global data quality distribution  
    quality_counts = count_conditions(assignments, 'quality')
    assert quality_counts == {'Q0': 45, 'Q1': 15, 'Q2': 15, 'Q3': 15}
    
    # Check 4: Individual participant balance
    for pid, tasks in assignments.items():
        assert len(tasks) == 10, f"Participant {pid} has {len(tasks)} tasks"
        
    # Check 5: Pattern representation
    validate_pattern_representation(assignments)
    
    return True

Post-Assignment Verification

Quality Metrics:

Condition distribution variance across participants
Order randomization effectiveness
Pattern adherence verification

