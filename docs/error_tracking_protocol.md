## docs/error_tracking_protocol.md

```markdown
# Error Tracking Protocol

## Overview

This document specifies the comprehensive error tracking and validation protocol for data quality manipulation in the manufacturing data assistant research. The protocol ensures systematic error injection, precise tracking, and reliable validation of data corruption effects.

## Data Quality Conditions

### Q0: Baseline (Perfect Data)
**Corruption Level**: 0%
**Description**: Unmodified manufacturing data with complete traceability
**Purpose**: Establishes ground truth and maximum performance baseline

**Characteristics**:
- All barcode formats strictly compliant
- Complete relationship linkages
- Full location tracking coverage
- Consistent timestamps across systems
- All documents properly formatted

### Q1: Space Injection Errors
**Corruption Level**: 15% of entries
**Target**: Barcode and ID fields
**Error Type**: Systematic space insertion, with white space inserted either at the front and/or back of the data field

**Implementation Details**:
Original: "3DOR10001"
Corrupted: " 3DOR10001", "3DOR10001  ", " 3DOR10001 "
Original: "1677565722"
Corrupted: "  1677565722", "1677565722 ", " 1677565722 "
Original: "ORBOX0011"
Corrupted: "ORBOX0011  ", "  ORBOX0011", " ORBOX0011 "

**Injection Strategy**:
- Random selection of 15% of barcode fields per dataset
- 1-3 spaces inserted at either star, end or both positions of ID strings
- Maintain recognizability for fuzzy matching validation

### Q2: Character Missing Errors  
**Corruption Level**: 12% of entries
**Target**: Gear IDs and Order IDs specifically
**Error Type**: Single character removal

**Implementation Details**:
Gear IDs:
Original: "3DOR10001" -> Corrupted: "3DR10001", "3DO10001", "3DOR1001"
Order IDs:
Original: "ORBOX0011" -> Corrupted: "RBOX0011", "OROX0011", "ORBOX011"

**Removal Strategy**:
- Random character position selection
- Single character deletion per affected entry
- Avoid removing leading characters that break format recognition

### Q3: Missing Records Errors
**Corruption Level**: 5-8% of records
**Target**: Relationship data and location data
**Error Type**: Strategic record deletion

**Implementation Details**:

**Priority Removal Targets**:
1. **Intermediate relationship links** (not breaking complete chains)
2. **Buffer area location scans** (Job End Buffer, Parts Warehouse)
3. **Secondary worker-printer associations**
4. **Redundant location tracking entries**

**Preservation Strategy**:
- Maintain at least one solution path per task
- Preserve essential printer-gear connections
- Keep order-gear final relationships intact
- Ensure baseline task solvability

## Error Tracking Implementation

### Dual Output File System

**Naming Convention**:

Original: location_data.csv
Corrupted: location_data_Q1.csv
Error Log: location_data_Q1_errors.csv

**File Pair Requirements**:
- Every corrupted dataset has corresponding error tracking file
- Consistent naming pattern for automated validation
- Cross-referenced indexing for researcher verification

### Error Log Schema

#### Q1 (Space Injection) Error Log
```csv
row,column,original,corrupted,error_type,timestamp
0,worker_id,1677565722,"167 7565722",Q1_SPACE,2024-11-15T10:30:00
15,gear_id,3DOR10001,"3D OR10001",Q1_SPACE,2024-11-15T10:30:01
23,order_id,ORBOX0011,"OR BOX0011",Q1_SPACE,2024-11-15T10:30:02

#### Q2 (Character Missing) Error Log
row,column,original,corrupted,error_type,removed_char,position
5,gear_id,3DOR10001,3DR10001,Q2_CHAR_MISSING,O,2
12,order_id,ORBOX0011,RBOX0011,Q2_CHAR_MISSING,O,0
18,gear_id,3DOR10002,3DOR1002,Q2_CHAR_MISSING,0,6

#### Q3 (Missing Records) Error Log
row,removed_record,error_type,affected_relationships,impact_assessment
45,"{""parent"":""3DOR10001"",""child"":""ORBOX0011""}",Q3_MISSING_RECORD,gear_to_order,MEDIUM
67,"{""location"":""Job End Buffer"",""id"":""3DOR10023""}",Q3_MISSING_RECORD,location_tracking,LOW
89,"{""worker"":""1677565722"",""printer"":""Printer_5""}",Q3_MISSING_RECORD,worker_assignment,LOW

In-File Error Annotations

CSV Comment Integration:
# ERROR_METADATA: Q1_SPACE_INJECTION, corruption_rate=0.15, timestamp=2024-11-15T10:30:00
# AFFECTED_ROWS: 0,15,23,34,45,56,67,78,89,92
# VALIDATION_HASH: a1b2c3d4e5f6g7h8i9j0
worker_id,location,timestamp,state
167 7565722,Goods In,2024-10-28T13:00:38.257018Z,entered  # ERROR: Q1_SPACE at position 3
1677565722,Printer Setup,2024-10-28T13:01:05.476953Z,entered
3D OR10001,Job End Buffer,2024-10-28T13:38:21.067950Z,entered  # ERROR: Q1_SPACE at position 2

Validation Framework

Automated Validation Checks

Pre-Injection Validation:

def validate_source_data(dataset_path):
    """Validate source data integrity before corruption"""
    
    # Check 1: Barcode format compliance
    validate_barcode_formats(dataset_path)
    
    # Check 2: Relationship completeness
    validate_relationship_integrity(dataset_path)
    
    # Check 3: Temporal consistency
    validate_timestamps(dataset_path)
    
    # Check 4: Cross-system references
    validate_cross_references(dataset_path)
    
    return validation_report

def validate_barcode_formats(dataset_path):
    """Verify all barcodes match expected patterns"""
    patterns = {
        'worker_rfid': r'^***REMOVED***d{10}$',
        'printer': r'^Printer_***REMOVED***d+$',
        'gear': r'^3DOR***REMOVED***d{5,6}$',
        'order': r'^OR[A-Z0-9]+$',
        'material': r'^[A-Z]{3,4}***REMOVED***d{4}$'
    }
    # Validation logic implementation

Post-Injection Validation:

def validate_error_injection(original_path, corrupted_path, error_log_path):
    """Validate error injection accuracy and completeness"""
    
    # Check 1: Error count matches target corruption rate
    validate_corruption_rate(original_path, corrupted_path, error_log_path)
    
    # Check 2: All errors properly logged
    validate_error_logging_completeness(corrupted_path, error_log_path)
    
    # Check 3: Error types match specifications
    validate_error_type_compliance(error_log_path)
    
    # Check 4: Task solvability preservation (Q3 only)
    if 'Q3' in corrupted_path:
        validate_task_solvability(corrupted_path)
    
    return validation_report

Manual Validation ProtocolResearcher Verification Process

Part 1: Statistical Validation

Corruption Rate Verification: Confirm actual vs. target corruption percentages
Distribution Analysis: Ensure random error distribution across dataset
Error Type Compliance: Verify errors match Q1/Q2/Q3 specifications

Part 2: Functional Validation

Task Solvability Check: Manually attempt task completion with corrupted data
Alternative Path Validation: Verify alternative solution routes exist for Q3
Error Detection Validation: Confirm errors are detectable but not trivial

Part 3: Cross-System Validation

Relationship Integrity: Check that essential connections remain intact
Temporal Consistency: Verify corruption doesn't break time sequences
Manufacturing Logic: Ensure corrupted data maintains operational realism

Error Classification System

Severity Assessment

Low Impact Errors:
Affect redundant data points
Alternative data sources available
Minimal impact on task completion
Example: Missing secondary location scan

Medium Impact Errors:
Affect primary data sources
Alternative paths require additional steps
Moderate increase in task difficulty
Example: Missing character in frequently-used gear ID

High Impact Errors:
Affect critical relationship links
Limited alternative solutions available
Significant increase in task complexity
Example: Missing printer-gear relationship for unique part

Critical Impact Errors (Avoided):
Break all solution paths
Make tasks unsolvable
Compromise experimental validity
Example: Deleting all location data for specific order
Quality Assurance MetricsCorruption Quality Indicators

Q1 Space Injection:
Target Rate: 15% ± 2%
Distribution: Uniform across barcode types
Detectability: >90% detectable by fuzzy matching algorithms
Recoverability: >95% recoverable with confidence >0.8

Q2 Character Missing:
Target Rate: 12% ± 2%
Focus: Gear and Order IDs only
Detectability: >85% detectable by pattern matching
Recoverability: >80% recoverable with moderate confidence

Q3 Missing Records:
Target Rate: 5-8% strategic removal
Preservation: 100% task solvability maintained
Alternative Paths: ≥1 solution route per task
Impact Distribution: 70% low, 25% medium, 5% high impact
Validation Success Criteria

Acceptance Thresholds:
Corruption rate within ±2% of target
Error logging completeness >99%
Task solvability preservation 100% (Q3)
Cross-validation consistency >95%
Manufacturing realism maintained

Rejection Criteria:
Any task becomes unsolvable
Corruption rate exceeds ±3% variance
Error logging incomplete or inconsistent
Manufacturing logic violations detected

Implementation Tools

Error Injection Pipeline

class DataQualityController:
    def __init__(self, corruption_specs):
        self.corruption_specs = corruption_specs
        self.error_tracker = ErrorTracker()
        
    def process_dataset(self, input_path, quality_condition):
        """Process dataset through quality condition pipeline"""
        
        # Step 1: Validate input data
        validation_report = self.validate_source_data(input_path)
        
        # Step 2: Apply corruption
        corrupted_data, error_log = self.apply_corruption(
            input_path, quality_condition
        )
        
        # Step 3: Generate output files
        output_paths = self.generate_output_files(
            corrupted_data, error_log, quality_condition
        )
        
        # Step 4: Validate corruption
        self.validate_corruption(input_path, output_paths)
        
        return output_paths

Error Analysis Tools

class ErrorAnalyzer:
    def analyze_corruption_impact(self, original_data, corrupted_data, tasks):
        """Analyze impact of corruption on task completion"""
        
        impact_analysis = {
            'task_completion_rates': {},
            'alternative_path_usage': {},
            'error_detection_rates': {},
            'confidence_degradation': {}
        }
        
        for task in tasks:
            # Simulate task completion under both conditions
            baseline_result = self.simulate_task(task, original_data)
            corrupted_result = self.simulate_task(task, corrupted_data)
            
            # Analyze performance differences
            impact_analysis['task_completion_rates'][task.id] = {
                'baseline': baseline_result.success_rate,
                'corrupted': corrupted_result.success_rate,
                'degradation': baseline_result.success_rate - corrupted_result.success_rate
            }
        
        return impact_analysis

