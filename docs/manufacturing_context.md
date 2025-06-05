## docs/manufacturing_context.md

````markdown
# Manufacturing Context Documentation

## 3D Printing Factory Environment

### Overview

This document provides detailed context for the manufacturing environment used in the LLM-based data assistant research. The scenario is based on a realistic 3D printing factory producing aerospace components with stringent traceability and certification requirements.

## Factory Operations

### Manufacturing Process Flow

┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Goods In   │ -> │ Printer Setup│ -> │ 3D Printing │
│  (Materials)│    │   (Workers)  │    │  (Machines) │
└─────────────┘    └──────────────┘    └─────────────┘
│                  │                  │
v                  v                  v
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Goods Out  │ <- │    Order     │ <- │ Job End     │
│ (Shipping)  │    │   Packing    │    │  Buffer     │
└─────────────┘    └──────────────┘    └─────────────┘
^                  ^                  │
│                  │                  v
└──────────────────┴───────────────────────┐
| Parts Warehouse                          |
│  (Storage)                               │ 
└───────────────────────────────────────────

### Process Stations

1. **Goods In**: Material receipt and initial barcode scanning
2. **Printer Setup**: Worker assignment and job preparation
3. **3D Printing**: Automated manufacturing with machine logging
4. **Job End Buffer**: Temporary storage post-printing
5. **Parts Warehouse**: Quality-controlled component storage
6. **Order Packing**: Final assembly and documentation
7. **Goods Out**: Shipping and final traceability confirmation

## Manufacturing Equipment

### 3D Printer Fleet

**Configuration**: 10 industrial-grade 3D printers
- **Printer IDs**: Printer_1 through Printer_10
- **Technology**: Selective Laser Sintering (SLS)
- **Materials**: Aerospace-grade thermoplastics (ABS, PEEK)
- **API Integration**: Real-time job logging with start/end timestamps

**Typical Job Characteristics**:
- Duration: 6 seconds to 35+ minutes depending on part complexity
- Material consumption: 0.1-2.0 kg per job
- Quality control: Automated dimensional verification

### Tracking Systems

#### Barcode Scanning Infrastructure
- **Coverage**: All 7 process stations equipped with industrial scanners
- **Technology**: 2D matrix barcodes with error correction
- **Integration**: Real-time updates to manufacturing execution system (MES)

#### RFID Worker Tracking
- **Worker Cards**: Unique 10-digit identifiers (e.g., 1677565722, 2199003780)
- **Activities Tracked**: 
  - Setting Up (job preparation)
  - Removing Build (post-printing part removal)
  - Taking Build Off (quality inspection)
- **Purpose**: Task accountability and labor cost tracking

## Data Architecture

### Manufacturing Data Sources

#### 1. Location Data System
**Purpose**: Real-time asset tracking through manufacturing process

**Data Elements**:
- Entity ID (barcodes for all tracked items)
- Location identifier (process station)
- Timestamp (entry/exit times)
- Status (entered/exited/in-process)

**Tracked Entities**:
- Materials: ABSM0002, ABSM0003, etc.
- Gears: 3DOR10001 to 3DOR100099
- Orders: ORBOX0011, ORBOX00111, etc.
- Workers: RFID card numbers
- Equipment: Printer_1, Printer_2, etc.

#### 2. Machine Log System
**Purpose**: Automated 3D printer operation recording

**Data Elements**:
- Machine identifier
- Job start timestamp
- Job end timestamp  
- Total duration (seconds)
- Material consumption
- Part identification

**Integration**: Direct API connection to printer controllers for real-time logging

#### 3. Relationship Database
**Purpose**: Entity associations and parent-child linkages

**Relationship Types**:
- Worker ↔ Printer assignments
- Printer ↔ Gear manufacturing
- Gear ↔ Order assembly
- Order ↔ Material consumption
- All relationships stored bidirectionally

#### 4. Worker Activity Logs
**Purpose**: RFID-based task tracking and labor analytics

**Activity Categories**:
- **Setting Up**: Job preparation, material loading, printer configuration
- **Removing Build**: Post-print part extraction and initial inspection
- **Taking Build Off**: Final quality checks and buffer placement

#### 5. Document Management System
**Purpose**: Certification and compliance documentation

**Document Types**:
- **FAA 8130-3 Certificates**: Airworthiness release documents
- **Packing Lists**: Order contents and shipping documentation
- **Quality Reports**: Inspection and test results

## Manufacturing Context Challenges

### Data Quality Issues

#### Real-World Manufacturing Data Problems
1. **Barcode Scanning Errors**:
   - Environmental factors (dust, lighting, wear)
   - Human error in scanning procedures
   - Equipment calibration drift

2. **System Integration Gaps**:
   - Network connectivity interruptions
   - Database synchronization delays
   - Legacy system compatibility issues

3. **Process Compliance Variations**:
   - Worker training consistency
   - Shift handover communication
   - Emergency procedure documentation

### Traceability Requirements

#### Aerospace Industry Standards
- **FAA Compliance**: Part 21 manufacturing certification
- **AS9100**: Quality management for aerospace
- **NADCAP**: Special process certification
- **ISO 9001**: General quality management

#### Critical Traceability Elements
1. **Material Lineage**: Raw material batch tracking
2. **Process Parameters**: Manufacturing condition recording
3. **Quality Verification**: Inspection and test documentation
4. **Worker Accountability**: Task-level responsibility tracking
5. **Equipment Qualification**: Machine capability validation

## Operational Queries

### Manufacturing Information Needs

#### Production Management
- "Which gears are in Order X for shipment planning?"
- "What's the status of parts in the buffer area?"
- "How many parts has Printer Y produced this week?"

#### Quality Control
- "Are all certifications complete for Product Z?"
- "Do completion dates match shipping schedules?"
- "Which worker performed final inspection on Part A?"

#### Maintenance and Efficiency
- "Which printer has the highest utilization rate?"
- "Are there any bottlenecks in the process flow?"
- "What's the average job duration by part type?"

#### Compliance and Auditing
- "Is traceability complete for this aerospace order?"
- "Do we have proper documentation for regulatory inspection?"
- "Are all required certifications dated correctly?"

## Worker Personas

### Manufacturing Roles

#### Production Operators
- **Primary Tasks**: Equipment setup, material handling, quality checks
- **Data Interaction**: Barcode scanning, RFID logging, basic status queries
- **Information Needs**: Job assignments, material locations, quality requirements

#### Production Supervisors  
- **Primary Tasks**: Schedule management, resource allocation, problem resolution
- **Data Interaction**: Multi-system queries, trend analysis, exception reporting
- **Information Needs**: Production status, bottleneck identification, worker productivity

#### Quality Inspectors
- **Primary Tasks**: Certification verification, compliance validation, documentation
- **Data Interaction**: Complex traceability queries, cross-system validation
- **Information Needs**: Complete part history, certification status, timeline verification

#### Maintenance Technicians
- **Primary Tasks**: Equipment servicing, calibration, troubleshooting
- **Data Interaction**: Machine performance analysis, utilization tracking
- **Information Needs**: Equipment history, performance trends, maintenance schedules

## Technology Integration

### Manufacturing Execution System (MES)
- **Platform**: Custom integration with SAP Manufacturing
- **Real-time Integration**: API connections to all tracking systems
- **Data Warehouse**: Historical data retention for analytics

### Enterprise Resource Planning (ERP)
- **Platform**: SAP S/4HANA with manufacturing modules
- **Integration Points**: Order management, inventory control, financial tracking
- **Reporting**: Standard manufacturing KPIs and compliance reports

### Quality Management System (QMS)
- **Platform**: MasterControl for document management
- **Certification Tracking**: FAA 8130-3 and AS9100 compliance
- **Integration**: Automated data feeds from manufacturing systems

This manufacturing context provides the realistic industrial environment necessary for evaluating LLM-based data assistant systems under authentic operational conditions while maintaining rigorous experimental control.