# src/config.py

DATA_PATH = "data/SCMS_Delivery_History_Dataset.csv"

DATE_COLS = [
    "PQ First Sent to Client Date",
    "PO Sent to Vendor Date",
    "Scheduled Delivery Date",
    "Delivered to Client Date",
    "Delivery Recorded Date",
]

CATEGORICAL_COLS = [
    "Country",
    "Managed By",
    "Fulfill Via",
    "Vendor INCO Term",
    "Shipment Mode",
    "Product Group",
    "Sub Classification",
    "Vendor",
    "Molecule/Test Type",
    "Brand",
    "Dosage Form",
    "Manufacturing Site",
    "First Line Designation",
]

NUMERIC_COLS_RAW = [
    "Unit of Measure (Per Pack)",
    "Line Item Quantity",
    "Line Item Value",
    "Pack Price",
    "Unit Price",
    "Weight (Kilograms)",
    "Freight Cost (USD)",
    "Line Item Insurance (USD)",
]
