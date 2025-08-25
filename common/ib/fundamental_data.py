#!/usr/bin/env python3
"""
IB Fundamental Data and Calendar Fetcher

Fetches fundamental data and calendar information from Interactive Brokers API.
Provides earnings calendars, financial ratios, company fundamentals, and WSH data.

Features:
- Company fundamental data (financial ratios, balance sheet, income statement)
- Earnings calendar from IB
- Analyst estimates and recommendations
- Corporate actions and calendar events
- WSH (What's Happening) data
- Real-time fundamental snapshots

Usage:
    from common.ib.fundamental_data import IBFundamentalDataFetcher

    fetcher = IBFundamentalDataFetcher()
    fundamentals = fetcher.get_company_fundamentals("AAPL")
    calendar = fetcher.get_earnings_calendar(["AAPL", "GOOGL"])
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Import IB API components
try:
    import threading

    from ibapi.client import EClient
    from ibapi.common import TickerId
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper

    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("⚠️  IB API not available - install with: pip install ibapi")

from common.utils.logging_setup import get_logger

logger = get_logger(__name__)


class FundamentalDataType(Enum):
    """Types of fundamental data available from IB."""

    FINANCIAL_SUMMARY = "ReportSnapshot"  # Key financial ratios
    COMPANY_OVERVIEW = "ReportsFinSummary"  # Company overview and ratios
    RATIOS = "ReportRatios"  # Financial ratios
    ESTIMATES = "CalendarReport"  # Analyst estimates and calendar
    OWNERSHIP = "ReportsOwnership"  # Institutional ownership
    ANALYST_FORECAST = "ReportsFinStatements"  # Analyst forecasts


@dataclass
class CompanyFundamentals:
    """Company fundamental data from IB."""

    symbol: str
    company_name: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps_ttm: Optional[float] = None
    revenue_ttm: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    beta: Optional[float] = None
    shares_outstanding: Optional[int] = None

    # Metadata
    data_date: Optional[date] = None
    source: str = "interactive_brokers"


@dataclass
class EarningsEstimate:
    """Earnings estimate data from IB."""

    symbol: str
    period_ending: Optional[date] = None
    period_type: str = "quarterly"  # quarterly, annual
    eps_estimate_mean: Optional[float] = None
    eps_estimate_high: Optional[float] = None
    eps_estimate_low: Optional[float] = None
    eps_estimate_count: Optional[int] = None
    revenue_estimate_mean: Optional[float] = None
    revenue_estimate_high: Optional[float] = None
    revenue_estimate_low: Optional[float] = None

    # Actual data (if available)
    eps_actual: Optional[float] = None
    revenue_actual: Optional[float] = None

    # Metadata
    last_updated: Optional[datetime] = None
    source: str = "interactive_brokers"


@dataclass
class CalendarEvent:
    """Calendar event from IB (earnings, dividends, splits, etc.)."""

    symbol: str
    event_type: str  # earnings, dividend, split, spinoff, etc.
    event_date: Optional[date] = None
    description: str = ""

    # Event-specific data
    dividend_amount: Optional[float] = None
    split_ratio: Optional[str] = None
    earnings_time: Optional[str] = None  # BMO, AMC, DMT

    # Metadata
    source: str = "interactive_brokers"


class IBFundamentalWrapper(EWrapper):
    """IB API Wrapper for fundamental data requests."""

    def __init__(self):
        super().__init__()
        self.fundamental_data = {}
        self.errors = {}
        self.request_complete = {}
        self.next_valid_id = 1

    def nextValidId(self, orderId: int):
        """Receive next valid order ID."""
        self.next_valid_id = orderId
        logger.debug(f"Next valid ID: {orderId}")

    def error(
        self,
        reqId: TickerId,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson="",
    ):
        """Handle errors."""
        self.errors[reqId] = {
            "code": errorCode,
            "message": errorString,
            "advanced": advancedOrderRejectJson,
        }
        self.request_complete[reqId] = True
        logger.error(f"IB Error {errorCode} for request {reqId}: {errorString}")

    def fundamentalData(self, reqId: TickerId, data: str):
        """Receive fundamental data."""
        self.fundamental_data[reqId] = data
        self.request_complete[reqId] = True
        logger.debug(
            f"Received fundamental data for request {reqId}: {len(data)} chars"
        )


class IBFundamentalClient(EClient):
    """IB API Client for fundamental data."""

    def __init__(self, wrapper):
        super().__init__(wrapper)


class IBFundamentalDataFetcher:
    """Fetches fundamental data and calendar information from IB API."""

    def __init__(self, host: str | None = None, port: int | None = None, client_id: int | None = None):
        """
        Initialize IB fundamental data fetcher.

        Args:
            host: IB Gateway/TWS host (required by caller or defaults to 127.0.0.1)
            port: IB Gateway/TWS port (required by caller or defaults to 4002)
            client_id: Client ID (required by caller or defaults to 200)
        """

        if not IB_AVAILABLE:
            raise ImportError("IB API not available. Install with: pip install ibapi")

        self.host = host or "127.0.0.1"
        self.port = int(port or 4002)
        self.client_id = int(client_id or 200)

        self.wrapper = IBFundamentalWrapper()
        self.client = IBFundamentalClient(self.wrapper)
        self.connected = False

        logger.info(f"📊 IB Fundamental Data Fetcher initialized")
        logger.info(f"   Host: {self.host}:{self.port}")
        logger.info(f"   Client ID: {self.client_id}")

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to IB Gateway/TWS."""
        try:
            logger.info(f"🔗 Connecting to IB Gateway at {self.host}:{self.port}")
            self.client.connect(self.host, self.port, self.client_id)

            # Start the client in a separate thread
            thread = threading.Thread(target=self.client.run)
            thread.daemon = True
            thread.start()

            # Wait for connection
            start_time = time.time()
            while (
                not self.client.isConnected() and (time.time() - start_time) < timeout
            ):
                time.sleep(0.1)

            if self.client.isConnected():
                self.connected = True
                logger.info("✅ Connected to IB Gateway")
                time.sleep(1)  # Give time for nextValidId
                return True
            else:
                logger.error("❌ Failed to connect to IB Gateway")
                return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from IB Gateway/TWS."""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            logger.info("🔌 Disconnected from IB Gateway")

    def _make_stock_contract(self, symbol: str, exchange: str = "SMART") -> Contract:
        """Create stock contract for IB API."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = "USD"
        return contract

    def _wait_for_data(self, req_id: int, timeout: float = 30.0) -> Optional[str]:
        """Wait for fundamental data response."""
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            if req_id in self.wrapper.request_complete:
                if req_id in self.wrapper.errors:
                    error = self.wrapper.errors[req_id]
                    logger.error(f"Request {req_id} failed: {error['message']}")
                    return None

                if req_id in self.wrapper.fundamental_data:
                    return self.wrapper.fundamental_data[req_id]

            time.sleep(0.1)

        logger.error(f"Request {req_id} timed out after {timeout}s")
        return None

    def get_company_fundamentals(self, symbol: str) -> Optional[CompanyFundamentals]:
        """
        NOTE: Fundamental data requires expensive IB data subscriptions.
        This method is disabled until proper XML parsing is implemented.

        Args:
            symbol: Stock symbol

        Returns:
            None (disabled)
        """
        logger.warning(
            f"📊 Fundamental data disabled for {symbol} - requires expensive IB subscriptions and proper XML parsing"
        )
        return None

    def get_earnings_calendar(self, symbols: List[str]) -> List[CalendarEvent]:
        """
        Get earnings calendar events for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            List of CalendarEvent objects
        """

        if not self.connected:
            if not self.connect():
                return []

        logger.info(f"📅 Fetching earnings calendar for {len(symbols)} symbols")

        calendar_events = []

        for symbol in symbols:
            try:
                contract = self._make_stock_contract(symbol)
                req_id = self.wrapper.next_valid_id
                self.wrapper.next_valid_id += 1

                # Request calendar data (includes earnings)
                self.client.reqFundamentalData(
                    req_id, contract, FundamentalDataType.ESTIMATES.value, []
                )

                # Wait for response
                data = self._wait_for_data(req_id)

                if data:
                    events = self._parse_calendar_data(symbol, data)
                    calendar_events.extend(events)
                else:
                    logger.warning(f"No calendar data received for {symbol}")

                # Rate limiting
                time.sleep(0.5)  # 2 requests per second

            except Exception as e:
                logger.error(f"Failed to fetch calendar for {symbol}: {e}")
                continue

        logger.info(f"📅 Retrieved {len(calendar_events)} calendar events")
        return calendar_events

    def get_earnings_estimates(self, symbol: str) -> List[EarningsEstimate]:
        """
        NOTE: Earnings estimates require expensive IB data subscriptions.
        This method is disabled until proper XML parsing is implemented.

        Args:
            symbol: Stock symbol

        Returns:
            Empty list (disabled)
        """
        logger.warning(
            f"📈 Earnings estimates disabled for {symbol} - requires expensive IB subscriptions"
        )
        return []

    def get_multiple_fundamentals(
        self, symbols: List[str]
    ) -> Dict[str, CompanyFundamentals]:
        """
        NOTE: Fundamental data disabled - requires expensive IB subscriptions.

        Args:
            symbols: List of stock symbols

        Returns:
            Empty dictionary (disabled)
        """
        logger.warning(
            f"📊 Multiple fundamentals disabled for {len(symbols)} symbols - requires expensive IB subscriptions"
        )
        return {}

    def _parse_fundamental_data(
        self, symbol: str, xml_data: str
    ) -> CompanyFundamentals:
        """Parse XML fundamental data from IB."""

        # Note: IB returns XML data that needs parsing
        # This is a simplified parser - in practice you'd use xml.etree.ElementTree

        fundamentals = CompanyFundamentals(
            symbol=symbol,
            company_name=symbol,  # Would extract from XML
            data_date=date.today(),
        )

        try:
            # Basic parsing - in real implementation, parse XML properly
            if "MarketCap" in xml_data:
                # Extract market cap from XML (simplified)
                pass

            if "PE" in xml_data or "PriceEarningsRatio" in xml_data:
                # Extract P/E ratio from XML
                pass

            # Note: This is a placeholder - real implementation would
            # properly parse the XML response from IB
            logger.debug(f"Parsed fundamental data for {symbol}")

        except Exception as e:
            logger.error(f"Failed to parse fundamental data for {symbol}: {e}")

        return fundamentals

    def _parse_calendar_data(self, symbol: str, xml_data: str) -> List[CalendarEvent]:
        """Parse earnings calendar from IB XML data focusing on BMO/AMC timing."""

        events = []

        if not xml_data:
            logger.warning(f"No XML data received for {symbol}")
            return events

        try:
            import xml.etree.ElementTree as ET

            logger.debug(f"Raw XML data for {symbol}: {xml_data[:500]}...")

            # Parse the XML
            root = ET.fromstring(xml_data)

            # Look for earnings events in various XML structures
            earnings_found = False

            # Search for earnings elements with different possible structures
            for elem in root.iter():
                if elem.tag and "earnings" in elem.tag.lower():
                    earnings_found = True

                    # Extract earnings date
                    event_date = None
                    for date_attr in ["date", "reportDate", "earningsDate"]:
                        if date_attr in elem.attrib:
                            try:
                                event_date = datetime.strptime(
                                    elem.attrib[date_attr], "%Y-%m-%d"
                                ).date()
                                break
                            except ValueError:
                                continue

                    # Extract earnings timing (BMO/AMC)
                    earnings_time = None
                    for time_attr in ["time", "timing", "marketTime"]:
                        if time_attr in elem.attrib:
                            timing = elem.attrib[time_attr].upper()
                            if "BEFORE" in timing or "BMO" in timing:
                                earnings_time = "BMO"
                            elif "AFTER" in timing or "AMC" in timing:
                                earnings_time = "AMC"
                            elif "DURING" in timing or "DMT" in timing:
                                earnings_time = "DMT"
                            break

                    # Look in element text for timing info
                    if not earnings_time and elem.text:
                        text = elem.text.upper()
                        if "BEFORE MARKET" in text or "BMO" in text:
                            earnings_time = "BMO"
                        elif "AFTER MARKET" in text or "AMC" in text:
                            earnings_time = "AMC"
                        elif "DURING MARKET" in text or "DMT" in text:
                            earnings_time = "DMT"

                    # Create calendar event
                    event = CalendarEvent(
                        symbol=symbol,
                        event_type="earnings",
                        event_date=event_date,
                        earnings_time=earnings_time,
                        description=f"Earnings for {symbol}"
                        + (f" ({earnings_time})" if earnings_time else ""),
                    )
                    events.append(event)

            # If no specific earnings elements, look for any date patterns
            if not earnings_found:
                # Look for any elements with date attributes that might be earnings
                for elem in root.iter():
                    if elem.attrib:
                        for attr_name, attr_value in elem.attrib.items():
                            if (
                                "date" in attr_name.lower()
                                and len(attr_value) == 10
                                and "-" in attr_value
                            ):
                                try:
                                    event_date = datetime.strptime(
                                        attr_value, "%Y-%m-%d"
                                    ).date()
                                    # Only consider future dates as potential earnings
                                    if event_date >= date.today():
                                        event = CalendarEvent(
                                            symbol=symbol,
                                            event_type="earnings",
                                            event_date=event_date,
                                            description=f"Potential earnings for {symbol}",
                                        )
                                        events.append(event)
                                        break
                                except ValueError:
                                    continue

            logger.info(f"📅 Parsed {len(events)} calendar events for {symbol}")
            for event in events:
                logger.info(
                    f"   {event.symbol}: {event.event_date} {event.earnings_time or 'TBD'}"
                )

        except ET.ParseError as e:
            logger.error(f"XML parsing failed for {symbol}: {e}")
            logger.debug(f"Problematic XML: {xml_data[:200]}")
        except Exception as e:
            logger.error(f"Failed to parse calendar data for {symbol}: {e}")

        return events

    def _parse_estimates_data(
        self, symbol: str, xml_data: str
    ) -> List[EarningsEstimate]:
        """Parse earnings estimates from IB XML data."""

        estimates = []

        try:
            # Parse XML for earnings estimates
            # This is a placeholder - real implementation would parse XML

            estimate = EarningsEstimate(
                symbol=symbol,
                period_ending=date.today() + timedelta(days=30),
                eps_estimate_mean=2.50,  # Placeholder values
                revenue_estimate_mean=1000000000,
                last_updated=datetime.now(),
            )
            estimates.append(estimate)

            logger.debug(f"Parsed {len(estimates)} estimates for {symbol}")

        except Exception as e:
            logger.error(f"Failed to parse estimates data for {symbol}: {e}")

        return estimates

    def export_fundamentals_to_csv(
        self, fundamentals: Dict[str, CompanyFundamentals], output_file: str = None
    ) -> str:
        """Export fundamental data to CSV file."""

        if not output_file:
            output_file = f"ib_fundamentals_{date.today().isoformat()}.csv"

        data = []
        for symbol, fund in fundamentals.items():
            data.append(asdict(fund))

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        logger.info(f"💾 Exported fundamentals to {output_file}")
        return output_file

    def export_calendar_to_csv(
        self, calendar_events: List[CalendarEvent], output_file: str = None
    ) -> str:
        """Export calendar events to CSV file."""

        if not output_file:
            output_file = f"ib_calendar_{date.today().isoformat()}.csv"

        data = [asdict(event) for event in calendar_events]
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        logger.info(f"💾 Exported calendar to {output_file}")
        return output_file


def main():
    """Test the IB fundamental data fetcher."""
    from common.utils.logging_setup import setup_logging

    setup_logging()

    print("📊 IB FUNDAMENTAL DATA FETCHER TEST")
    print("=" * 60)

    if not IB_AVAILABLE:
        print("❌ IB API not available")
        print("💡 Install with: pip install ibapi")
        return

    fetcher = IBFundamentalDataFetcher()

    try:
        # Test connection
        if not fetcher.connect():
            print("❌ Failed to connect to IB Gateway")
            print("💡 Make sure IB Gateway or TWS is running")
            return

        print("✅ Connected to IB Gateway")

        # Test symbols
        test_symbols = ["AAPL", "GOOGL", "MSFT"]

        print(f"\n📊 Testing fundamental data for {test_symbols}")

        # Get fundamentals for one symbol
        print(f"\n1️⃣ Getting fundamentals for AAPL...")
        aapl_fundamentals = fetcher.get_company_fundamentals("AAPL")

        if aapl_fundamentals:
            print(f"✅ AAPL fundamentals retrieved:")
            print(f"   Company: {aapl_fundamentals.company_name}")
            print(f"   Market Cap: {aapl_fundamentals.market_cap}")
            print(f"   P/E Ratio: {aapl_fundamentals.pe_ratio}")
            print(f"   Data Date: {aapl_fundamentals.data_date}")
        else:
            print("❌ Failed to get AAPL fundamentals")

        # Get earnings calendar
        print(f"\n2️⃣ Getting earnings calendar for {test_symbols}...")
        calendar_events = fetcher.get_earnings_calendar(test_symbols)

        if calendar_events:
            print(f"✅ Found {len(calendar_events)} calendar events:")
            for event in calendar_events[:5]:  # Show first 5
                print(f"   {event.symbol}: {event.event_type} on {event.event_date}")
        else:
            print("❌ No calendar events found")

        # Get earnings estimates
        print(f"\n3️⃣ Getting earnings estimates for AAPL...")
        estimates = fetcher.get_earnings_estimates("AAPL")

        if estimates:
            print(f"✅ Found {len(estimates)} estimates:")
            for est in estimates:
                print(f"   Period: {est.period_ending}")
                print(f"   EPS Estimate: {est.eps_estimate_mean}")
                print(f"   Revenue Estimate: {est.revenue_estimate_mean}")
        else:
            print("❌ No estimates found")

        # Get multiple fundamentals
        print(f"\n4️⃣ Getting fundamentals for multiple symbols...")
        multiple_fundamentals = fetcher.get_multiple_fundamentals(
            test_symbols[:2]
        )  # Test with 2 symbols

        if multiple_fundamentals:
            print(
                f"✅ Retrieved fundamentals for {len(multiple_fundamentals)} symbols:"
            )
            for symbol, fund in multiple_fundamentals.items():
                print(
                    f"   {symbol}: {fund.company_name} (Market Cap: {fund.market_cap})"
                )
        else:
            print("❌ No fundamentals retrieved")

        # Export data
        if multiple_fundamentals:
            print(f"\n💾 Exporting data...")
            fund_file = fetcher.export_fundamentals_to_csv(multiple_fundamentals)
            print(f"   Fundamentals: {fund_file}")

        if calendar_events:
            cal_file = fetcher.export_calendar_to_csv(calendar_events)
            print(f"   Calendar: {cal_file}")

        print(f"\n🎯 Usage Examples:")
        print(f"   • Compare IB fundamental data with external sources")
        print(f"   • Cross-validate earnings dates from different sources")
        print(f"   • Get real-time analyst estimates and recommendations")
        print(f"   • Track corporate actions and calendar events")
        print(f"   • Integrate with earnings discovery for comprehensive analysis")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    finally:
        fetcher.disconnect()
        print(f"\n🔌 Disconnected from IB Gateway")


if __name__ == "__main__":
    main()
