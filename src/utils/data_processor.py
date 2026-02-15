"""
Data Processor Module - DuckDB Integration Layer

Handles CSV loading, data validation, and SQL query execution for the Retail Insights Assistant.
"""

import os
import duckdb
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from .input_loader import load_dataframe_from_bytes, infer_table_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    DuckDB-based data processor for handling retail sales data.
    
    Provides methods for:
    - Loading CSV files into DuckDB
    - Executing SQL queries
    - Schema introspection
    - Data validation
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize DataProcessor with DuckDB connection.
        
        Args:
            db_path: Path to DuckDB database file (":memory:" for in-memory)
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_connection()
        self.loaded_tables = []
        self._loaded_directories = set()  # Track loaded directories to avoid reloading
    
    def _initialize_connection(self):
        """Initialize DuckDB connection."""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"DuckDB connection established: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")
            raise
    
    def load_csv(self, file_path: str, table_name: Optional[str] = None) -> bool:
        """
        Load a CSV file into DuckDB.
        
        Args:
            file_path: Path to CSV file
            table_name: Name for the table (defaults to filename without extension)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Generate table name if not provided
            if table_name is None:
                table_name = Path(file_path).stem.replace(" ", "_").replace("-", "_")
            
            # Read CSV and register with DuckDB
            df = pd.read_csv(file_path)
            self.conn.register(table_name, df)
            if table_name not in self.loaded_tables:
                self.loaded_tables.append(table_name)
            
            logger.info(f"Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            return False
    
    def load_uploaded_file(self, file_name: str, file_bytes: bytes, table_name: Optional[str] = None) -> bool:
        """Load an uploaded file (CSV/Excel/JSON/TXT) into DuckDB."""
        try:
            if table_name is None:
                table_name = infer_table_name(file_name)

            df = load_dataframe_from_bytes(file_name, file_bytes)
            if df is None or df.empty:
                logger.warning(f"Loaded file {file_name} is empty")
                return False

            self.conn.register(table_name, df)
            if table_name not in self.loaded_tables:
                self.loaded_tables.append(table_name)

            logger.info(f"Loaded {file_name} into table {table_name}: {len(df)} rows")
            return True
        except Exception as e:
            logger.error(f"Failed to load uploaded file {file_name}: {e}")
            return False

    def load_report_file(self, file_path: str, table_name: Optional[str] = None) -> bool:
        """Load local report file (CSV/Excel/JSON/TXT) into DuckDB."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return False

            if table_name is None:
                table_name = infer_table_name(path.name)

            return self.load_uploaded_file(path.name, path.read_bytes(), table_name)
        except Exception as e:
            logger.error(f"Failed to load report file {file_path}: {e}")
            return False

    def load_all_csvs(self, directory: str) -> Dict[str, bool]:
        """
        Load all CSV files from a directory.
        Skips loading if already loaded from this directory.
        
        Args:
            directory: Path to directory containing CSV files
            
        Returns:
            Dictionary mapping filename to load status
        """
        # Avoid reloading the same directory
        if directory in self._loaded_directories:
            logger.info(f"Directory already loaded: {directory}. Skipping.")
            return {}
        
        results = {}
        
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return results
        
        csv_files = list(Path(directory).glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {directory}")
        
        for csv_file in csv_files:
            results[csv_file.name] = self.load_csv(str(csv_file))
        
        # Mark directory as loaded
        self._loaded_directories.add(directory)
        
        return results
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query string
            
        Returns:
            Pandas DataFrame with query results
        """
        try:
            result = self.conn.execute(sql).fetch_df()
            logger.info(f"Query executed successfully: {len(result)} rows returned")
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary mapping column names to data types
        """
        try:
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            schema = {row[0]: row[1] for row in result}
            return schema
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {e}")
            return {}
    
    def get_numeric_columns(self, table_name: str) -> List[str]:
        """
        Get list of numeric columns in a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of numeric column names
        """
        schema = self.get_table_schema(table_name)
        numeric_types = {"BIGINT", "INTEGER", "SMALLINT", "TINYINT", "DOUBLE", "FLOAT", "DECIMAL"}
        return [col for col, dtype in schema.items() if any(nt in dtype for nt in numeric_types)]
    
    def get_text_columns(self, table_name: str) -> List[str]:
        """
        Get list of text columns in a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of text column names
        """
        schema = self.get_table_schema(table_name)
        text_types = {"VARCHAR", "STRING", "TEXT"}
        return [col for col, dtype in schema.items() if any(tt in dtype for tt in text_types)]
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get basic statistics for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with row count and column information
        """
        try:
            row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            schema = self.get_table_schema(table_name)
            
            stats = {
                "row_count": row_count,
                "column_count": len(schema),
                "columns": schema,
                "numeric_columns": self.get_numeric_columns(table_name),
                "text_columns": self.get_text_columns(table_name),
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for {table_name}: {e}")
            return {}
    
    def list_tables(self) -> List[str]:
        """
        Get list of all loaded tables.
        
        Returns:
            List of table names
        """
        try:
            result = self.conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
            tables = [row[0] for row in result]
            return tables
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        Get list of columns for a specific table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            List of column names
        """
        try:
            result = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            columns = [row[1] for row in result]  # row[1] is the column name in PRAGMA output
            return columns
        except Exception as e:
            logger.error(f"Failed to get columns for table {table_name}: {e}")
            return []
    
    def get_sample_data(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """
        Get sample rows from a table.
        
        Args:
            table_name: Name of the table
            limit: Number of rows to retrieve
            
        Returns:
            Pandas DataFrame with sample data
        """
        try:
            return self.query(f"SELECT * FROM {table_name} LIMIT {limit}")
        except Exception as e:
            logger.error(f"Failed to get sample data from {table_name}: {e}")
            return pd.DataFrame()
    
    def validate_table(self, table_name: str) -> Dict[str, Any]:
        """
        Validate table structure and content.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Validation report
        """
        try:
            stats = self.get_table_stats(table_name)
            schema = self.get_table_schema(table_name)
            
            # Check for null values
            null_counts = {}
            for col in schema.keys():
                null_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
                if null_count > 0:
                    null_counts[col] = null_count
            
            validation_report = {
                "table_name": table_name,
                "is_valid": True,
                "row_count": stats.get("row_count", 0),
                "column_count": stats.get("column_count", 0),
                "null_columns": null_counts,
                "schema": schema,
            }
            
            return validation_report
        except Exception as e:
            logger.error(f"Validation failed for {table_name}: {e}")
            return {"is_valid": False, "error": str(e)}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global processor instance
_processor_instance = None


def get_processor(db_path: str = ":memory:") -> DataProcessor:
    """
    Get or create the global DataProcessor instance.
    
    Args:
        db_path: Path to DuckDB database (defaults to in-memory)
        
    Returns:
        DataProcessor instance
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DataProcessor(db_path)
    return _processor_instance


def reset_processor():
    """Reset the global processor instance."""
    global _processor_instance
    if _processor_instance:
        _processor_instance.close()
    _processor_instance = None


if __name__ == "__main__":
    # Example usage
    processor = get_processor(":memory:")
    
    # Load sample CSV
    csv_path = "data/Sales Dataset/Amazon Sale Report.csv"
    if os.path.exists(csv_path):
        processor.load_csv(csv_path)
        
        # Get statistics
        tables = processor.list_tables()
        print(f"Loaded tables: {tables}")
        
        for table in tables:
            stats = processor.get_table_stats(table)
            print(f"\nTable: {table}")
            print(f"  Rows: {stats.get('row_count', 0)}")
            print(f"  Columns: {stats.get('column_count', 0)}")
    
    processor.close()
