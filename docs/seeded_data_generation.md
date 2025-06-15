# Seeded Data Generation for Reproducible Experiments

## Overview

The manufacturing data assistant now uses **seeded random number generation** to ensure that data corruption patterns are completely reproducible. This is crucial for debugging, validation, and ensuring consistent experimental results.

## How It Works

### 1. Random Seed Configuration

The random seed is configured in `config/experiment_config.yaml`:

```yaml
experiment:
  random_seed: 42  # Fixed seed for reproducible corruption patterns
```

### 2. Seeded Data Quality Controller

The `DataQualityController` class now accepts a `random_seed` parameter:

```python
# Create controller with seeded randomness
controller = DataQualityController(random_seed=42)

# Apply corruption - will be identical every time with same seed
corrupted_data, error_tracker, targeted_ids = controller.apply_corruption("Q1")
```

### 3. Quality-Specific Seeds

Different quality conditions use different seeds to ensure varied corruption patterns:

- **Q1 (Space Injection)**: `base_seed + 1000`
- **Q2 (Character Removal)**: `base_seed + 2000` 
- **Q3 (Record Removal)**: `base_seed + 3000`

This ensures that Q1, Q2, and Q3 have different corruption patterns while remaining reproducible.

## Benefits

### ✅ **Reproducible Debugging**
- Same seed always produces identical corruption patterns
- Debugging sessions can be replicated exactly
- Error patterns remain consistent across runs

### ✅ **Experimental Consistency**
- LLM evaluation results are comparable across runs
- Ground truth validation produces consistent results
- Error identification accuracy can be measured reliably

### ✅ **Version Control Friendly**
- Error log files won't change unexpectedly
- Git diffs show only intentional changes
- Collaboration is easier with predictable data

## Usage

### Regenerating Datasets with Seeds

Use the dedicated script to regenerate all datasets with seeded corruption:

```bash
python scripts/regenerate_seeded_datasets.py
```

This script will:
1. ✅ Verify seed consistency
2. 🔄 Backup existing datasets
3. 🌱 Regenerate Q1, Q2, Q3 with seeded corruption
4. 📊 Provide detailed summary

### Running Main Generation Pipeline

The main generation scripts now automatically use seeded generation:

```bash
python run_phase1_generation.py
```

### Testing with Seeds

Tests use fixed seeds for reproducible results:

```python
def test_q1_space_injection():
    ctrl = DataQualityController(random_seed=0)
    data, tracker, targeted_ids = ctrl.apply_corruption("Q1")
    # Results will be identical every time
```

## Current Seed Configuration

**Base Seed**: `42` (configured in `experiment_config.yaml`)

**Effective Seeds**:
- Q1: `1042` (42 + 1000)
- Q2: `2042` (42 + 2000)  
- Q3: `3042` (42 + 3000)

## Verification

The system includes built-in verification to ensure seed consistency:

```python
# Verify that same seed produces identical results
controller1 = DataQualityController(random_seed=42)
controller2 = DataQualityController(random_seed=42)

# Both should produce identical corruption patterns
result1 = controller1.apply_corruption("Q1")
result2 = controller2.apply_corruption("Q1")
assert result1 == result2  # ✅ Identical results
```

## Error Log Preservation

The original error logs from June 9th have been preserved and are now protected by the seeded generation system:

- **Original Timestamps**: `2025-06-09T19:18:32.*`
- **Original Patterns**: Preserved exactly as generated
- **Future Safety**: Won't be overwritten by accidental regeneration

## Migration Notes

### Before Seeding
- Random corruption patterns changed every run
- Debugging was difficult due to inconsistent data
- Error logs could be accidentally overwritten

### After Seeding  
- ✅ Corruption patterns are deterministic
- ✅ Debugging is reproducible
- ✅ Error logs are protected
- ✅ Experimental results are consistent

## Best Practices

1. **Don't Change the Seed**: Keep `random_seed: 42` unless you need completely new corruption patterns
2. **Use Regeneration Script**: Always use `regenerate_seeded_datasets.py` for controlled regeneration
3. **Backup Before Changes**: The script automatically backs up existing datasets
4. **Verify Consistency**: Run the built-in verification before important experiments
5. **Document Changes**: If you change the seed, document why and what changed

## Troubleshooting

### Problem: Different results with same seed
**Solution**: Ensure you're using the `DataQualityController(random_seed=X)` constructor, not manual `random.seed()`

### Problem: Need different corruption patterns
**Solution**: Change the `random_seed` value in `config/experiment_config.yaml` and regenerate

### Problem: Lost original error logs
**Solution**: Original logs are preserved in git history at commit `82efb0c` and can be restored

---

**Result**: The manufacturing data assistant now has fully reproducible, debuggable data generation with preserved experimental consistency! 🌱✅
