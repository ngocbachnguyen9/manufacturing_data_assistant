from .location_query_tool import LocationQueryTool
from .machine_log_tool import MachineLogTool
from .relationship_tool import RelationshipTool
from .worker_data_tool import WorkerDataTool
from .document_parser_tool import DocumentParserTool
from .barcode_validator_tool import BarcodeValidatorTool

# A dictionary to easily access all tool classes
TOOL_CLASSES = {
    "location_query_tool": LocationQueryTool,
    "machine_log_tool": MachineLogTool,
    "relationship_tool": RelationshipTool,
    "worker_data_tool": WorkerDataTool,
    "document_parser_tool": DocumentParserTool,
    "barcode_validator_tool": BarcodeValidatorTool,
}