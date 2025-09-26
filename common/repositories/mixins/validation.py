"""Validation functionality for repositories."""

import logging
import pandas as pd
import pyarrow as pa
from typing import List, Dict, Any, Optional, Callable, Union
from decimal import Decimal, InvalidOperation
from datetime import date, datetime

from ...exceptions import DataValidationException
from ...constants import MAX_SYMBOL_LENGTH, MIN_SYMBOL_LENGTH

logger = logging.getLogger(__name__)


class ValidationMixin:
    """Adds data validation capabilities to repositories."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_enabled = True
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        
    def add_validation_rule(
        self,
        column: str,
        rule_type: str,
        **rule_params
    ) -> None:
        """Add a validation rule for a column.
        
        Args:
            column: Column name to validate
            rule_type: Type of validation (required, type, range, etc.)
            **rule_params: Additional parameters for the rule
        """
        if column not in self.validation_rules:
            self.validation_rules[column] = {}
        
        self.validation_rules[column][rule_type] = rule_params
        logger.debug(f"Added {rule_type} validation rule for column {column}")
    
    def add_custom_validator(
        self,
        name: str,
        validator_func: Callable[[Any], Union[bool, str]]
    ) -> None:
        """Add a custom validation function.
        
        Args:
            name: Name of the validator
            validator_func: Function that takes a value and returns True if valid,
                          or error message string if invalid
        """
        self.custom_validators[name] = validator_func
        logger.debug(f"Added custom validator: {name}")
    
    def validate_data(
        self,
        data: pd.DataFrame,
        raise_on_error: bool = True,
        **kwargs
    ) -> List[str]:
        """Validate DataFrame against defined rules.
        
        Args:
            data: DataFrame to validate
            raise_on_error: Whether to raise exception on validation failure
            **kwargs: Additional validation parameters
            
        Returns:
            List of validation error messages (empty if valid)
            
        Raises:
            DataValidationException: If validation fails and raise_on_error=True
        """
        if not self.validation_enabled:
            return []
        
        errors = []
        
        # Validate against schema if available
        if hasattr(self, 'get_schema'):
            try:
                schema = self.get_schema()
                schema_errors = self._validate_schema(data, schema)
                errors.extend(schema_errors)
            except Exception as e:
                errors.append(f"Schema validation failed: {e}")
        
        # Apply column-specific validation rules
        for column, rules in self.validation_rules.items():
            column_errors = self._validate_column(data, column, rules)
            errors.extend(column_errors)
        
        # Apply custom validators
        for validator_name, validator_func in self.custom_validators.items():
            try:
                result = validator_func(data)
                if result is not True:
                    if isinstance(result, str):
                        errors.append(f"Custom validator {validator_name}: {result}")
                    else:
                        errors.append(f"Custom validator {validator_name} failed")
            except Exception as e:
                errors.append(f"Custom validator {validator_name} error: {e}")
        
        # Domain-specific validations
        domain_errors = self._validate_domain_specific(data, **kwargs)
        errors.extend(domain_errors)
        
        if errors and raise_on_error:
            error_msg = f"Data validation failed with {len(errors)} errors: " + "; ".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"... and {len(errors) - 5} more"
            raise DataValidationException(error_msg, details={'errors': errors})
        
        if errors:
            logger.warning(f"Data validation found {len(errors)} errors")
        else:
            logger.debug("Data validation passed")
        
        return errors
    
    def _validate_schema(self, data: pd.DataFrame, schema: pa.Schema) -> List[str]:
        """Validate DataFrame against PyArrow schema."""
        errors = []
        
        # Check required columns
        schema_columns = set(schema.names)
        data_columns = set(data.columns)
        
        missing_columns = schema_columns - data_columns
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check data types for existing columns
        for column in schema_columns.intersection(data_columns):
            try:
                expected_type = schema.field(column).type
                # Convert to pandas and back to check compatibility
                pa.Array.from_pandas(data[column], type=expected_type)
            except pa.ArrowTypeError as e:
                errors.append(f"Column {column} type mismatch: {e}")
            except Exception as e:
                errors.append(f"Column {column} validation error: {e}")
        
        return errors
    
    def _validate_column(
        self,
        data: pd.DataFrame,
        column: str,
        rules: Dict[str, Any]
    ) -> List[str]:
        """Validate a specific column against its rules."""
        errors = []
        
        if column not in data.columns:
            if 'required' in rules and rules['required']:
                errors.append(f"Required column {column} is missing")
            return errors
        
        series = data[column]
        
        # Required validation
        if rules.get('required', False):
            null_count = series.isnull().sum()
            if null_count > 0:
                errors.append(f"Column {column} has {null_count} null values (required)")
        
        # Type validation
        if 'type' in rules:
            expected_type = rules['type']
            type_errors = self._validate_column_type(series, column, expected_type)
            errors.extend(type_errors)
        
        # Range validation
        if 'min_value' in rules or 'max_value' in rules:
            range_errors = self._validate_column_range(
                series, column, 
                rules.get('min_value'),
                rules.get('max_value')
            )
            errors.extend(range_errors)
        
        # Length validation (for strings)
        if 'min_length' in rules or 'max_length' in rules:
            length_errors = self._validate_column_length(
                series, column,
                rules.get('min_length'),
                rules.get('max_length')
            )
            errors.extend(length_errors)
        
        # Pattern validation
        if 'pattern' in rules:
            pattern_errors = self._validate_column_pattern(series, column, rules['pattern'])
            errors.extend(pattern_errors)
        
        # Unique validation
        if rules.get('unique', False):
            duplicate_count = series.duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Column {column} has {duplicate_count} duplicate values (must be unique)")
        
        return errors
    
    def _validate_column_type(
        self,
        series: pd.Series,
        column: str,
        expected_type: Union[str, type]
    ) -> List[str]:
        """Validate column data type."""
        errors = []
        
        # Skip null values for type checking
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return errors
        
        if expected_type == 'numeric':
            if not pd.api.types.is_numeric_dtype(series):
                errors.append(f"Column {column} must be numeric")
        elif expected_type == 'string':
            if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
                errors.append(f"Column {column} must be string")
        elif expected_type == 'datetime':
            if not pd.api.types.is_datetime64_any_dtype(series):
                errors.append(f"Column {column} must be datetime")
        elif expected_type == 'date':
            # Check if values can be converted to dates
            try:
                pd.to_datetime(non_null_series.iloc[:min(10, len(non_null_series))]).dt.date
            except Exception:
                errors.append(f"Column {column} must contain valid dates")
        
        return errors
    
    def _validate_column_range(
        self,
        series: pd.Series,
        column: str,
        min_value: Optional[Union[int, float, Decimal]],
        max_value: Optional[Union[int, float, Decimal]]
    ) -> List[str]:
        """Validate column values are within specified range."""
        errors = []
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return errors
        
        try:
            if min_value is not None:
                below_min = (non_null_series < min_value).sum()
                if below_min > 0:
                    errors.append(f"Column {column} has {below_min} values below minimum {min_value}")
            
            if max_value is not None:
                above_max = (non_null_series > max_value).sum()
                if above_max > 0:
                    errors.append(f"Column {column} has {above_max} values above maximum {max_value}")
        except Exception as e:
            errors.append(f"Range validation error for column {column}: {e}")
        
        return errors
    
    def _validate_column_length(
        self,
        series: pd.Series,
        column: str,
        min_length: Optional[int],
        max_length: Optional[int]
    ) -> List[str]:
        """Validate string column length."""
        errors = []
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return errors
        
        try:
            lengths = non_null_series.astype(str).str.len()
            
            if min_length is not None:
                too_short = (lengths < min_length).sum()
                if too_short > 0:
                    errors.append(f"Column {column} has {too_short} values shorter than {min_length}")
            
            if max_length is not None:
                too_long = (lengths > max_length).sum()
                if too_long > 0:
                    errors.append(f"Column {column} has {too_long} values longer than {max_length}")
        except Exception as e:
            errors.append(f"Length validation error for column {column}: {e}")
        
        return errors
    
    def _validate_column_pattern(
        self,
        series: pd.Series,
        column: str,
        pattern: str
    ) -> List[str]:
        """Validate column values match regex pattern."""
        errors = []
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return errors
        
        try:
            import re
            compiled_pattern = re.compile(pattern)
            
            invalid_count = 0
            for value in non_null_series:
                if not compiled_pattern.match(str(value)):
                    invalid_count += 1
            
            if invalid_count > 0:
                errors.append(f"Column {column} has {invalid_count} values not matching pattern {pattern}")
        except Exception as e:
            errors.append(f"Pattern validation error for column {column}: {e}")
        
        return errors
    
    def _validate_domain_specific(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """Validate domain-specific business rules."""
        errors = []
        
        # Symbol validation
        if 'symbol' in data.columns:
            symbol_errors = self._validate_symbols(data['symbol'])
            errors.extend(symbol_errors)
        
        # Date validation
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        for col in date_columns:
            date_errors = self._validate_dates(data[col], col)
            errors.extend(date_errors)
        
        # Price validation
        price_columns = [col for col in data.columns if any(
            price_term in col.lower() 
            for price_term in ['price', 'strike', 'premium']
        )]
        for col in price_columns:
            price_errors = self._validate_prices(data[col], col)
            errors.extend(price_errors)
        
        # Volume validation
        if 'volume' in data.columns:
            volume_errors = self._validate_volume(data['volume'])
            errors.extend(volume_errors)
        
        return errors
    
    def _validate_symbols(self, symbol_series: pd.Series) -> List[str]:
        """Validate trading symbols."""
        errors = []
        
        non_null_symbols = symbol_series.dropna()
        if len(non_null_symbols) == 0:
            return errors
        
        # Check symbol length
        lengths = non_null_symbols.str.len()
        too_short = (lengths < MIN_SYMBOL_LENGTH).sum()
        too_long = (lengths > MAX_SYMBOL_LENGTH).sum()
        
        if too_short > 0:
            errors.append(f"Found {too_short} symbols shorter than {MIN_SYMBOL_LENGTH} characters")
        if too_long > 0:
            errors.append(f"Found {too_long} symbols longer than {MAX_SYMBOL_LENGTH} characters")
        
        # Check for invalid characters (only alphanumeric allowed)
        invalid_chars = non_null_symbols.str.contains(r'[^A-Z0-9]', regex=True).sum()
        if invalid_chars > 0:
            errors.append(f"Found {invalid_chars} symbols with invalid characters (only A-Z, 0-9 allowed)")
        
        return errors
    
    def _validate_dates(self, date_series: pd.Series, column: str) -> List[str]:
        """Validate date values."""
        errors = []
        
        non_null_dates = date_series.dropna()
        if len(non_null_dates) == 0:
            return errors
        
        # Try to convert to datetime to check validity
        try:
            converted_dates = pd.to_datetime(non_null_dates)
            
            # Check for future dates (usually not valid for historical data)
            future_dates = (converted_dates > pd.Timestamp.now()).sum()
            if future_dates > 0:
                errors.append(f"Column {column} has {future_dates} future dates")
            
            # Check for very old dates (before 1900)
            very_old = (converted_dates < pd.Timestamp('1900-01-01')).sum()
            if very_old > 0:
                errors.append(f"Column {column} has {very_old} dates before 1900")
        except Exception as e:
            errors.append(f"Date validation error for column {column}: {e}")
        
        return errors
    
    def _validate_prices(self, price_series: pd.Series, column: str) -> List[str]:
        """Validate price values."""
        errors = []
        
        non_null_prices = price_series.dropna()
        if len(non_null_prices) == 0:
            return errors
        
        # Check for negative prices
        negative_prices = (non_null_prices < 0).sum()
        if negative_prices > 0:
            errors.append(f"Column {column} has {negative_prices} negative values")
        
        # Check for extremely high prices (likely errors)
        extremely_high = (non_null_prices > 100000).sum()
        if extremely_high > 0:
            errors.append(f"Column {column} has {extremely_high} values above $100,000 (possibly invalid)")
        
        # Check for zero prices (often invalid)
        zero_prices = (non_null_prices == 0).sum()
        if zero_prices > 0:
            errors.append(f"Column {column} has {zero_prices} zero values")
        
        return errors
    
    def _validate_volume(self, volume_series: pd.Series) -> List[str]:
        """Validate volume values."""
        errors = []
        
        non_null_volume = volume_series.dropna()
        if len(non_null_volume) == 0:
            return errors
        
        # Check for negative volume
        negative_volume = (non_null_volume < 0).sum()
        if negative_volume > 0:
            errors.append(f"Volume has {negative_volume} negative values")
        
        # Check for non-integer volume
        non_integer = (non_null_volume % 1 != 0).sum()
        if non_integer > 0:
            errors.append(f"Volume has {non_integer} non-integer values")
        
        return errors
    
    def get_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get current validation rules."""
        return self.validation_rules.copy()
    
    def set_validation_enabled(self, enabled: bool) -> None:
        """Enable or disable validation."""
        self.validation_enabled = enabled
        logger.info(f"Validation {'enabled' if enabled else 'disabled'}")