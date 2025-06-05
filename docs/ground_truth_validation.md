## docs/ground_truth_validation.md

````markdown
# Ground Truth Validation

## Overview

This document outlines the comprehensive framework for establishing, validating, and maintaining ground truth answers for the manufacturing data assistant research. The ground truth serves as the definitive standard for evaluating both human and LLM performance across all task complexity levels and data quality conditions.

## Ground Truth Philosophy

### Definition and Scope

**Ground Truth**: The definitive, correct answers to manufacturing queries derived from perfect (Q0) baseline data through systematic data traversal methods that reflect realistic factory operations.

**Key Principles**:
1. **Manufacturing Realism**: Answers must reflect actual factory operational requirements
2. **Data Traversal Validity**: Solution paths must follow logical manufacturing data relationships  
3. **Expert Validation**: All answers verified by manufacturing domain experts
4. **Reproducibility**: Answer generation process must be fully documented and repeatable

### Answer Categories by Task Complexity

#### Easy Tasks (Direct Traversal)
**Query Type**: "Find all gears for Order X"
**Ground Truth Elements**:
- Complete list of gear IDs associated with the order
- Verification that gears reached completion status
- Data traversal path documentation

#### Medium Tasks (Cross-System Validation)  
**Query Type**: "Determine printer for Part Y and count parts on that printer"
**Ground Truth Elements**:
- Correct printer identification
- Accurate part count for that printer
- Cross-system validation pathway

#### Hard Tasks (Document Cross-Reference)
**Query Type**: "Verify ARC document dates vs warehouse arrival"
**Ground Truth Elements**:
- Certificate completion date extraction
- Warehouse arrival timestamp identification
- Date match/discrepancy determination
- Compliance assessment

## Ground Truth Generation Methodology

### Baseline Data Analysis (Q0)

#### Systematic Data Traversal Process

**Step 1: Relationship Mapping**
```python
def map_manufacturing_relationships(baseline_data):
    """Create comprehensive relationship map from perfect data"""
    
    relationship_graph = {
        'worker_printer': {},      # Worker assignments to printers
        'printer_gear': {},        # Printer production of gears
        'gear_order': {},          # Gear inclusion in orders
        'order_material': {},      # Material consumption by orders
        'location_tracking': {}    # Movement through process stations
    }
    
    # Build bidirectional relationship mappings
    for relationship in baseline_data['relationship_data']:
        parent = relationship['parent']
        child = relationship['child']
        
        # Store both directions for efficient lookup
        add_relationship(relationship_graph, parent, child)
        add_relationship(relationship_graph, child, parent)
    
    return relationship_graph


**Step 2: Location Timeline Construction**

def construct_location_timelines(baseline_data):
    """Build temporal sequences for all tracked entities"""
    
    timelines = {}
    for scan in baseline_data['location_data']:
        entity_id = scan['id']
        timestamp = scan['timestamp']
        location = scan['location']
        state = scan['state']
        
        if entity_id not in timelines:
            timelines[entity_id] = []
        
        timelines[entity_id].append({
            'timestamp': timestamp,
            'location': location,
            'state': state
        })
    
    # Sort by timestamp for each entity
    for entity_id in timelines:
        timelines[entity_id].sort(key=lambda x: x['timestamp'])
    
    return timelines

**Step 3: Document Cross-Reference**

def extract_document_data(baseline_data):
    """Extract and cross-reference certificate information"""
    
    document_registry = {}
    for cert_file in baseline_data['certificates']:
        cert_data = parse_faa_certificate(cert_file)
        
        document_registry[cert_data['serial_number']] = {
            'completion_date': cert_data['completion_date'],
            'part_number': cert_data['part_number'],
            'organization': cert_data['organization'],
            'certificate_number': cert_data['certificate_number']
        }
    
    return document_registry

Answer Generation Algorithm

Easy Task Ground Truth Generation:

def generate_easy_ground_truth(order_id, baseline_data):
    """Generate ground truth for gear identification tasks"""
    
    # Step 1: Find all gears associated with the order
    relationships = baseline_data['relationship_data']
    order_gears = find_children(relationships, order_id, type_filter='gear')
    
    # Step 2: Verify gear completion status
    location_data = baseline_data['location_data']
    completed_gears = []
    
    for gear_id in order_gears:
        timeline = get_location_timeline(location_data, gear_id)
        completion_status = check_completion_status(timeline)
        
        completed_gears.append({
            'gear_id': gear_id,
            'status': completion_status,
            'final_location': timeline[-1]['location'] if timeline else 'UNKNOWN'
        })
    
    return {
        'order_id': order_id,
        'total_gears': len(completed_gears),
        'gear_list': completed_gears,
        'data_traversal_path': ['relationship_data', 'location_data'],
        'success_criteria': 'All gears listed with completion status'
    }

Medium Task Ground Truth Generation:

def generate_medium_ground_truth(part_id, baseline_data):
    """Generate ground truth for printer identification and counting"""
    
    # Step 1: Find printer for the specific part
    relationships = baseline_data['relationship_data']
    part_printer = find_parent(relationships, part_id, type_filter='printer')
    
    # Step 2: Count all parts printed on that printer
    machine_logs = baseline_data['machine_log']
    printer_parts = []
    
    for log_entry in machine_logs:
        if log_entry['machine'] == part_printer:
            # Find associated part through relationships
            associated_parts = find_parts_for_job(
                relationships, log_entry, part_printer
            )
            printer_parts.extend(associated_parts)
    
    # Step 3: Cross-validate with location data
    location_validation = validate_printer_assignment(
        baseline_data['location_data'], part_id, part_printer
    )
    
    return {
        'part_id': part_id,
        'assigned_printer': part_printer,
        'total_parts_on_printer': len(set(printer_parts)),
        'validation_status': location_validation,
        'data_traversal_path': ['relationship_data', 'machine_log', 'location_data'],
        'success_criteria': 'Correct printer ID and accurate part count'
    }

Hard Task Ground Truth Generation:

def generate_hard_ground_truth(product_id, baseline_data):
    """Generate ground truth for document-location cross-reference"""
    
    # Step 1: Extract certificate completion date
    certificates = baseline_data['certificates']
    cert_data = find_certificate_for_product(certificates, product_id)
    completion_date = cert_data['completion_date']
    
    # Step 2: Find warehouse arrival date
    location_data = baseline_data['location_data']
    warehouse_timeline = get_location_timeline(location_data, product_id)
    warehouse_entry = find_warehouse_arrival(warehouse_timeline)
    arrival_date = warehouse_entry['timestamp']
    
    # Step 3: Compare dates and assess compliance
    date_match = compare_dates(completion_date, arrival_date)
    compliance_status = assess_compliance_impact(date_match)
    
    return {
        'product_id': product_id,
        'certificate_date': completion_date,
        'warehouse_arrival_date': arrival_date,
        'date_match_status': date_match['status'],
        'date_discrepancy_days': date_match['discrepancy_days'],
        'compliance_assessment': compliance_status,
        'data_traversal_path': ['certificates', 'location_data', 'relationship_data'],
        'success_criteria': 'Accurate date comparison and compliance assessment'
    }

Error Condition Ground Truth

Data Quality Impact on Ground Truth

Q1 (Space Injection) Ground Truth Modifications

Requirements for Correct Answers:
Primary Answer: Same as baseline ground truth
Error Detection: Must identify space injection errors
Confidence Assessment: Lower confidence due to data quality issues
Recovery Method: Document fuzzy matching or space removal techniques

Q2 (Character Missing) Ground Truth Modifications

Requirements for Correct Answers:
Primary Answer: Same as baseline with noted uncertainty
Error Detection: Must identify missing character patterns
Alternative Approaches: Document alternative data sources used
Confidence Degradation: Quantified confidence reduction

Q3 (Missing Records) Ground Truth Modifications

Requirements for Correct Answers:
Primary Answer: May be partial due to missing data
Error Detection: Must identify missing relationship/location records
Alternative Paths: Document alternative solution routes taken
Impact Assessment: Evaluate manufacturing process implications

Expert Validation Framework

Manufacturing Domain Expert Review

Validation Panel Composition
Lead Manufacturing Engineer: Process flow and operational accuracy
Quality Assurance Manager: Compliance and certification requirements
MES System Administrator: Data integration and technical accuracy
Production Supervisor: Factory floor operational realism

Validation Criteria

Operational Accuracy:
Do answers reflect realistic factory operations?
Are data traversal paths operationally logical?
Do timing expectations match manufacturing realities?

Technical Precision:
Are barcode formats and relationships accurate?
Do machine logs align with printer capabilities?
Are document formats compliant with FAA standards?

Compliance Validity:
Do certification requirements match aerospace standards?
Are traceability paths sufficient for regulatory compliance?
Do date validations reflect actual compliance procedures?

Validation Process

Phase 1: Individual Expert Review

Each expert independently reviews:
Sample ground truth answers across all complexity levels
Data traversal pathway documentation
Manufacturing realism assessment
Technical accuracy verification

Phase 2: Expert Panel Consensus

Collaborative review session to:
Resolve disagreements between individual reviews
Establish final ground truth validation
Document any modifications or clarifications needed
Create final approved ground truth dataset

Phase 3: Ongoing Validation

Continuous validation process:
Periodic review of experimental results for ground truth accuracy
Adjustment of answers if manufacturing context understanding evolves
Documentation of any ground truth modifications during study

Ground Truth Storage and Management

Data Structure

Ground Truth Database Schema

{
  "task_id": "easy_ORBOX0011_001",
  "complexity_level": "easy",
  "query_template": "Find all gears for Order {order_id}",
  "query_instance": "Find all gears for Order ORBOX0011",
  "baseline_answer": {
    "order_id": "ORBOX0011",
    "gear_count": 10,
    "gear_list": ["3DOR10012", "3DOR10013", "..."],
    "completion_status": "all_completed"
  },
  "data_traversal_path": ["relationship_data", "location_data"],
  "success_criteria": "All gears identified with completion verification",
  "data_quality_modifications": {
    "Q1": {
      "error_detection_required": true,
      "confidence_adjustment": -0.15,
      "additional_requirements": ["space_error_identification"]
    },
    "Q2": {
      "error_detection_required": true,
      "confidence_adjustment": -0.25,
      "additional_requirements": ["character_missing_identification"]
    },
    "Q3": {
      "error_detection_required": true,
      "confidence_adjustment": -0.35,
      "additional_requirements": ["missing_record_identification", "alternative_path_documentation"]
    }
  },
  "expert_validation": {
    "validated_by": ["expert_1", "expert_2", "expert_3"],
    "validation_date": "2024-11-15",
    "consensus_score": 0.95,
    "modifications_made": []
  }
}

Version Control and Traceability

Ground Truth Versioning:
Version 1.0: Initial expert-validated ground truth
Version 1.1: Minor corrections based on pilot testing
Version 1.2: Adjustments from human study insights

Change Documentation

All ground truth modifications documented with:
Reason for change
Expert approval
Impact on existing evaluations
Backward compatibility assessment

Quality Assurance

Automated Validation Checks

Consistency Verification

def validate_ground_truth_consistency(ground_truth_db):
    """Automated checks for ground truth internal consistency"""
    
    # Check 1: All referenced entities exist in baseline data
    validate_entity_references(ground_truth_db)
    
    # Check 2: Data traversal paths are logically sound
    validate_traversal_paths(ground_truth_db)
    
    # Check 3: Success criteria are measurable
    validate_success_criteria(ground_truth_db)
    
    # Check 4: Cross-task consistency
    validate_cross_task_consistency(ground_truth_db)
    
    return validation_report

Manufacturing Logic Validation

def validate_manufacturing_logic(ground_truth_db, baseline_data):
    """Verify answers align with manufacturing process constraints"""
    
    # Check temporal consistency
    validate_temporal_sequences(ground_truth_db, baseline_data)
    
    # Check process flow compliance
    validate_process_flow_adherence(ground_truth_db, baseline_data)
    
    # Check capacity constraints
    validate_manufacturing_capacity(ground_truth_db, baseline_data)
    
    return logic_validation_report

Manual Quality Assurance

Spot Check Protocol

Random sampling of 10% of ground truth answers
Independent verification by second expert
Cross-validation with actual manufacturing data patterns
Verification of data quality condition modifications

Regression Testing

Test ground truth against known manufacturing scenarios
Validate with historical factory data patterns
Ensure answers remain stable across methodology iterations