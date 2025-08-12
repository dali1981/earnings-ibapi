# Testing Guide for Trading Project

This directory contains comprehensive tests for the trading project, organized into multiple categories with proper fixtures, mocking, and utilities.

## Test Structure

```
tests/
├── conftest.py              # Global fixtures and configuration
├── pytest.ini              # Pytest configuration
├── requirements-test.txt    # Test-specific dependencies
├── test_runner.py          # Enhanced test runner script
├── unit/                   # Unit tests
│   ├── test_equity_bars_repository.py
│   ├── test_ib_core.py
│   └── ...
├── integration/            # Integration tests
│   ├── test_streaming.py
│   └── ...
├── data/                   # Data validation tests
│   ├── test_data_validation.py
│   └── ...
└── mocks/                  # Mock utilities
    ├── ib_mock_factory.py
    └── ...
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual components in isolation
- Fast execution (< 1s per test)
- Heavy use of mocking
- High code coverage focus

### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- Use real-like data flows
- May include file system operations
- Moderate execution time (1-10s per test)

### Data Tests (`@pytest.mark.data`)
- Validate data schemas and quality
- Test repository operations
- Check data transformation logic
- Focus on data integrity

### API Tests (`@pytest.mark.api`)
- Test IB API integration components
- Mock external API calls
- Verify request/response handling
- Test error scenarios

### Performance Tests (`@pytest.mark.slow`)
- Test with large datasets
- Memory usage validation
- Concurrent operation testing
- Longer execution times (10s+ per test)

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run specific category
pytest -m unit
pytest -m integration
pytest -m data
pytest -m api

# Run with coverage
pytest --cov=. --cov-report=html

# Run in parallel
pytest -n auto
```

### Using Test Runner

```bash
# Set up test environment
python tests/test_runner.py --setup

# Validate test structure
python tests/test_runner.py --validate

# Run specific test suites
python tests/test_runner.py --unit
python tests/test_runner.py --integration
python tests/test_runner.py --data

# Generate comprehensive report
python tests/test_runner.py --report

# Clean test artifacts
python tests/test_runner.py --clean
```

### Advanced Options

```bash
# Run tests with custom markers
pytest -m "unit and not slow"

# Run specific test file
pytest tests/unit/test_equity_bars_repository.py

# Run with verbose output
pytest -v

# Run with timeout
pytest --timeout=300

# Skip slow tests
pytest -m "not slow"
```

## Key Features

### Comprehensive Fixtures
- **Repository Fixtures**: Pre-configured repositories with test data
- **IB API Mocking**: Realistic IB API client mocks with data generation
- **Configuration**: Test-specific configuration objects
- **Data Generation**: Realistic market data for testing

### IB API Mocking
The `MockIBClientFactory` provides:
- Realistic market data generation
- Configurable delays and error rates
- Contract details simulation
- Historical and real-time data mocking

### Data Validation
- Pandera schemas for data validation
- Quality metrics calculation
- Consistency checks for OHLC data
- Performance validation for large datasets

### Error Testing
- Network error simulation
- Data corruption scenarios
- Timeout and retry logic testing
- Edge case handling

## Writing New Tests

### Unit Test Example
```python
@pytest.mark.unit
class TestMyComponent:
    def test_basic_functionality(self, mock_dependency):
        # Arrange
        component = MyComponent(mock_dependency)
        
        # Act
        result = component.process_data()
        
        # Assert
        assert result.success
        mock_dependency.method.assert_called_once()
```

### Integration Test Example
```python
@pytest.mark.integration
class TestDataFlow:
    def test_end_to_end_flow(self, temp_data_dir, mock_ib_client):
        # Test complete data flow
        streamer = create_streamer(mock_ib_client, temp_data_dir)
        streamer.run()
        
        # Verify data was persisted
        assert (temp_data_dir / "output.parquet").exists()
```

### Data Test Example
```python
@pytest.mark.data
class TestDataQuality:
    def test_schema_validation(self, sample_data):
        # Validate against Pandera schema
        equity_bar_schema.validate(sample_data)
```

## Best Practices

1. **Use Appropriate Markers**: Tag tests with correct markers
2. **Mock External Dependencies**: Use fixtures for IB API, file system, etc.
3. **Test Edge Cases**: Include error conditions and boundary cases
4. **Keep Tests Fast**: Optimize unit tests for speed
5. **Use Realistic Data**: Generate realistic test data for better coverage
6. **Clean Up**: Use fixtures for temporary resources
7. **Document Complex Tests**: Add docstrings for complex test scenarios

## Configuration

### pytest.ini
- Test discovery patterns
- Default markers
- Coverage settings
- Warning filters

### conftest.py
- Global fixtures
- Test configuration
- Mock setup
- Cleanup utilities

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example CI configuration
- name: Run Tests
  run: |
    python tests/test_runner.py --setup
    python tests/test_runner.py --validate
    python tests/test_runner.py --report
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project is in PYTHONPATH
2. **Fixture Not Found**: Check conftest.py imports
3. **Mock Issues**: Verify mock setup in fixtures
4. **Slow Tests**: Use `pytest -m "not slow"` to skip performance tests
5. **Coverage Issues**: Check for missing test files

### Debug Mode
```bash
# Run with debug output
pytest -vvv --tb=long

# Run single test with debugging
pytest -vvv tests/unit/test_example.py::TestClass::test_method
```

## Performance Considerations

- Unit tests should complete in < 1 second each
- Integration tests should complete in < 10 seconds each
- Use `@pytest.mark.slow` for tests that take longer
- Consider parallel execution for test suites
- Monitor memory usage in performance tests

## Test Data Management

- Use fixtures for consistent test data
- Generate realistic market data with MockDataGenerator
- Clean up temporary files automatically
- Version control test data schemas