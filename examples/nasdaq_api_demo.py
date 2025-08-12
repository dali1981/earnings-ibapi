#!/usr/bin/env python3
"""
NASDAQ Earnings Calendar API Demo

This script demonstrates how to access NASDAQ's public earnings calendar API
and shows the exact data structure and coverage available.

NASDAQ API Reference:
- Base URL: https://api.nasdaq.com/api/calendar/earnings
- Documentation: https://www.nasdaq.com/market-activity/earnings
- Web Interface: https://www.nasdaq.com/market-activity/earnings

The API provides free access to earnings calendar data without requiring
an API key, making it excellent for market-wide earnings discovery.
"""

import json
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

class NASDAQEarningsAPI:
    """
    NASDAQ Earnings Calendar API Client
    
    Provides access to NASDAQ's public earnings calendar data.
    Reference: https://www.nasdaq.com/market-activity/earnings
    """
    
    def __init__(self):
        self.base_url = "https://api.nasdaq.com/api/calendar/earnings"
        self.session = requests.Session()
        
        # Headers to mimic browser request (sometimes required)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nasdaq.com/',
            'Origin': 'https://www.nasdaq.com'
        })
        
    def get_earnings_for_date(self, target_date: date) -> Dict[str, Any]:
        """
        Get all earnings announcements for a specific date.
        
        Args:
            target_date: Date to fetch earnings for
            
        Returns:
            Dict with raw API response including metadata
        """
        params = {
            'date': target_date.strftime('%Y-%m-%d')
        }
        
        try:
            print(f"🔍 Fetching NASDAQ earnings for {target_date}...")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Add metadata about the request
            data['_request_info'] = {
                'date_requested': target_date.isoformat(),
                'api_url': response.url,
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return {'error': str(e)}
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            return {'error': f'JSON decode error: {e}'}
    
    def analyze_api_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure and content of API response."""
        
        analysis = {
            'api_info': {
                'endpoint': self.base_url,
                'documentation': 'https://www.nasdaq.com/market-activity/earnings',
                'web_interface': 'https://www.nasdaq.com/market-activity/earnings',
                'api_key_required': False,
                'rate_limits': 'Unknown - appears to be generous for reasonable usage'
            },
            'response_structure': {},
            'data_quality': {},
            'sample_records': [],
            'statistics': {}
        }
        
        if 'error' in response_data:
            analysis['error'] = response_data['error']
            return analysis
        
        # Analyze response structure
        analysis['response_structure'] = {
            'top_level_keys': list(response_data.keys()),
            'has_data_section': 'data' in response_data,
            'has_status': 'status' in response_data
        }
        
        if response_data.get('_request_info'):
            analysis['request_info'] = response_data['_request_info']
        
        # Analyze data section
        if 'data' in response_data and response_data['data']:
            data_section = response_data['data']
            analysis['data_structure'] = {
                'data_keys': list(data_section.keys()) if isinstance(data_section, dict) else 'not_dict',
                'has_rows': 'rows' in data_section if isinstance(data_section, dict) else False
            }
            
            # Analyze earnings records
            if isinstance(data_section, dict) and 'rows' in data_section:
                rows = data_section['rows']
                
                if rows and len(rows) > 0:
                    analysis['statistics'] = {
                        'total_companies': len(rows),
                        'record_fields': list(rows[0].keys()) if rows else []
                    }
                    
                    # Sample records (first 3)
                    analysis['sample_records'] = rows[:3]
                    
                    # Data quality analysis
                    symbols = [row.get('symbol', '') for row in rows]
                    company_names = [row.get('companyName', '') for row in rows]
                    
                    analysis['data_quality'] = {
                        'symbols_with_data': len([s for s in symbols if s]),
                        'companies_with_names': len([n for n in company_names if n]),
                        'unique_symbols': len(set(symbols)),
                        'sample_symbols': symbols[:10],
                        'symbol_patterns': self._analyze_symbol_patterns(symbols)
                    }
        
        return analysis
    
    def _analyze_symbol_patterns(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze patterns in stock symbols."""
        
        patterns = {
            'total_symbols': len(symbols),
            'valid_symbols': 0,
            'likely_stocks': 0,
            'likely_etfs': 0,
            'foreign_symbols': 0,
            'length_distribution': {}
        }
        
        length_counts = {}
        
        for symbol in symbols:
            if not symbol:
                continue
                
            symbol = symbol.strip().upper()
            patterns['valid_symbols'] += 1
            
            # Length analysis
            length = len(symbol)
            length_counts[length] = length_counts.get(length, 0) + 1
            
            # Pattern analysis
            if symbol.isalpha():
                if length <= 4:
                    patterns['likely_stocks'] += 1
                elif length == 3 and symbol.endswith(('ETF', 'VIX')):
                    patterns['likely_etfs'] += 1
            
            # Foreign/ADR detection (basic)
            if '.' in symbol or '-' in symbol or len(symbol) > 5:
                patterns['foreign_symbols'] += 1
        
        patterns['length_distribution'] = length_counts
        
        return patterns
    
    def demo_multiple_days(self, days: int = 7) -> List[Dict[str, Any]]:
        """Demo fetching earnings for multiple days."""
        
        results = []
        start_date = date.today()
        
        print(f"📅 NASDAQ API Demo - Fetching {days} days of earnings")
        print(f"{'='*60}")
        
        for i in range(days):
            target_date = start_date + timedelta(days=i)
            
            # Skip weekends (usually no earnings)
            if target_date.weekday() >= 5:
                continue
                
            response = self.get_earnings_for_date(target_date)
            analysis = self.analyze_api_response(response)
            
            results.append({
                'date': target_date,
                'response': response,
                'analysis': analysis
            })
            
            # Show summary
            if 'statistics' in analysis:
                companies = analysis['statistics'].get('total_companies', 0)
                print(f"✅ {target_date} ({target_date.strftime('%A')}): {companies} companies")
            else:
                print(f"❌ {target_date}: No data or error")
            
            # Be respectful to NASDAQ's servers
            import time
            time.sleep(0.5)
        
        return results
    
    def export_detailed_analysis(self, results: List[Dict[str, Any]], output_dir: Path = Path("nasdaq_analysis")):
        """Export detailed analysis of NASDAQ API data."""
        
        output_dir.mkdir(exist_ok=True)
        
        # 1. Summary report
        summary_file = output_dir / "nasdaq_api_summary.json"
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'api_endpoint': self.base_url,
            'documentation_links': [
                'https://www.nasdaq.com/market-activity/earnings',
                'https://api.nasdaq.com/api/calendar/earnings'
            ],
            'total_days_analyzed': len(results),
            'days_with_data': len([r for r in results if 'statistics' in r.get('analysis', {})]),
            'total_companies_found': sum(
                r['analysis'].get('statistics', {}).get('total_companies', 0) 
                for r in results
            ),
            'daily_breakdown': []
        }
        
        for result in results:
            analysis = result.get('analysis', {})
            summary['daily_breakdown'].append({
                'date': result['date'].isoformat(),
                'companies': analysis.get('statistics', {}).get('total_companies', 0),
                'has_error': 'error' in analysis
            })
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Summary exported to: {summary_file}")
        
        # 2. Raw data samples
        sample_file = output_dir / "nasdaq_raw_samples.json"
        
        samples = {
            'sample_responses': [],
            'field_analysis': {},
            'symbol_examples': []
        }
        
        all_symbols = []
        all_fields = set()
        
        for result in results:
            if 'sample_records' in result.get('analysis', {}):
                samples['sample_responses'].append({
                    'date': result['date'].isoformat(),
                    'sample_records': result['analysis']['sample_records']
                })
                
                # Collect field names
                for record in result['analysis']['sample_records']:
                    all_fields.update(record.keys())
                
                # Collect symbols
                symbols = result['analysis'].get('data_quality', {}).get('sample_symbols', [])
                all_symbols.extend(symbols)
        
        samples['field_analysis'] = {
            'all_fields_found': sorted(list(all_fields)),
            'field_descriptions': {
                'symbol': 'Stock ticker symbol',
                'companyName': 'Company name',
                'time': 'Earnings announcement time (bmo/amc/etc)',
                'consensus': 'Consensus earnings estimate',
                'noOfEsts': 'Number of analyst estimates'
            }
        }
        
        samples['symbol_examples'] = {
            'all_symbols_sample': sorted(list(set(all_symbols)))[:50],
            'total_unique_symbols': len(set(all_symbols))
        }
        
        with open(sample_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"📋 Raw samples exported to: {sample_file}")
        
        # 3. CSV export for spreadsheet analysis
        csv_file = output_dir / "nasdaq_earnings_data.csv"
        
        csv_data = []
        for result in results:
            target_date = result['date']
            response = result.get('response', {})
            
            if 'data' in response and 'rows' in response['data']:
                for row in response['data']['rows']:
                    csv_row = {
                        'date': target_date.isoformat(),
                        'symbol': row.get('symbol', ''),
                        'company_name': row.get('companyName', ''),
                        'time': row.get('time', ''),
                        'consensus': row.get('consensus', ''),
                        'no_of_estimates': row.get('noOfEsts', ''),
                        'market_cap': row.get('marketCap', ''),
                        'last_year_eps': row.get('lastYearEPS', ''),
                        'last_year_rpt_dt': row.get('lastYearRptDt', ''),
                        'surprise_percent': row.get('surprisePercent', '')
                    }
                    csv_data.append(csv_row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False)
            print(f"📈 CSV data exported to: {csv_file}")
        
        return output_dir


def main():
    """Demo the NASDAQ Earnings API."""
    
    print("🏛️ NASDAQ EARNINGS CALENDAR API DEMO")
    print("=" * 70)
    
    print(f"📚 API Information:")
    print(f"   Endpoint: https://api.nasdaq.com/api/calendar/earnings")
    print(f"   Documentation: https://www.nasdaq.com/market-activity/earnings")
    print(f"   Web Interface: https://www.nasdaq.com/market-activity/earnings")
    print(f"   API Key Required: No")
    print(f"   Rate Limits: Reasonable usage appears to be tolerated")
    
    # Initialize API client
    api = NASDAQEarningsAPI()
    
    # Demo single day
    print(f"\n📅 Single Day Demo:")
    today = date.today()
    response = api.get_earnings_for_date(today)
    analysis = api.analyze_api_response(response)
    
    if 'statistics' in analysis:
        stats = analysis['statistics']
        print(f"   Companies found today: {stats['total_companies']}")
        print(f"   Data fields available: {stats['record_fields']}")
        
        if 'sample_records' in analysis and analysis['sample_records']:
            print(f"\n   Sample earnings record:")
            sample = analysis['sample_records'][0]
            for key, value in sample.items():
                print(f"     {key}: {value}")
    
    # Demo multiple days
    print(f"\n📊 Multiple Days Demo:")
    results = api.demo_multiple_days(days=10)
    
    # Summary statistics
    total_companies = sum(
        r['analysis'].get('statistics', {}).get('total_companies', 0) 
        for r in results
    )
    
    days_with_data = len([
        r for r in results 
        if 'statistics' in r.get('analysis', {})
    ])
    
    print(f"\n📈 Summary:")
    print(f"   Total days analyzed: {len(results)}")
    print(f"   Days with earnings data: {days_with_data}")
    print(f"   Total companies found: {total_companies}")
    print(f"   Average companies per day: {total_companies / max(days_with_data, 1):.1f}")
    
    # Show data quality
    if results and 'data_quality' in results[0].get('analysis', {}):
        quality = results[0]['analysis']['data_quality']
        print(f"\n📊 Data Quality (sample day):")
        print(f"   Symbols with data: {quality.get('symbols_with_data', 0)}")
        print(f"   Companies with names: {quality.get('companies_with_names', 0)}")
        print(f"   Unique symbols: {quality.get('unique_symbols', 0)}")
        
        if 'symbol_patterns' in quality:
            patterns = quality['symbol_patterns']
            print(f"   Likely stocks: {patterns.get('likely_stocks', 0)}")
            print(f"   Foreign/ADR symbols: {patterns.get('foreign_symbols', 0)}")
    
    # Export detailed analysis
    print(f"\n💾 Exporting detailed analysis...")
    output_dir = api.export_detailed_analysis(results)
    
    print(f"\n🎯 Key Findings:")
    print(f"   • NASDAQ API provides free access to earnings calendar")
    print(f"   • Covers {total_companies} companies across {days_with_data} trading days")
    print(f"   • No API key required - public access")
    print(f"   • Data includes symbol, company name, timing, estimates")
    print(f"   • Suitable for market-wide earnings discovery")
    print(f"   • Rate limits appear generous for reasonable usage")
    
    print(f"\n📁 Analysis files created in: {output_dir.absolute()}")
    print(f"   • nasdaq_api_summary.json - Overview and statistics")
    print(f"   • nasdaq_raw_samples.json - Sample API responses")  
    print(f"   • nasdaq_earnings_data.csv - All earnings data")
    
    print(f"\n🔗 Reference Links:")
    print(f"   • API Endpoint: https://api.nasdaq.com/api/calendar/earnings")
    print(f"   • Web Interface: https://www.nasdaq.com/market-activity/earnings")
    print(f"   • NASDAQ Market Data: https://www.nasdaq.com/market-activity")


if __name__ == "__main__":
    main()