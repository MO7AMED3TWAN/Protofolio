# Data Flow Pipeline in Azure Synapse Analytics

This project demonstrates a Data Flow pipeline in **Azure Synapse Analytics**, where data is sourced from three different files (JSON, Excel, CSV), combined using a **Union transformation**, and stored in a final destination as a CSV file.

## Project Overview

The goal of this project is to load data from three different source formats, combine them into a single dataset, and store the results in a CSV file for further analysis or machine learning tasks.

## Steps in the Data Flow Pipeline:

### 1. Source Files:
- **JSON File**: The first data source is a JSON file containing structured data.
- **Excel File**: The second source is an Excel file with tabular data.
- **CSV File**: The third source is a CSV file with additional data.

### 2. Transformation:
- **Union Transformation**: A union transformation was applied to combine all three datasets into one. The union operation merged the data from these different sources into a single unified dataset.

### 3. Destination:
- **CSV File**: The final output data was written to a CSV file as the destination. This CSV file contains the merged data from the three sources.

## Key Components:

- **Azure Synapse Analytics**:
  - Used to build and manage the entire data pipeline.
  - Provides the ability to handle multiple data sources and apply transformations.
  
- **Data Flow**:
  - The core of the project that handles transformations and data movement between sources and destinations.

- **Union Transformation**:
  - Combines data from the three different formats (JSON, Excel, CSV) into a single dataset for output.
