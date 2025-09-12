'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { toast } from 'sonner';
import { api } from '@/lib/api';

interface Contract {
  symbol: string;
  expiry: string;
  strike: number;
  right: string;
  conId?: number;
  localSymbol?: string;
  exchange?: string;
  multiplier?: number;
}

interface ContractsResult {
  symbol: string;
  contract_count: number;
  contracts: Contract[];
}

interface OptionContractsComponentProps {
  initialActive?: 'select' | 'enrich' | 'save' | 'results' | 'historical' | 'preview';
}

export default function OptionContractsComponent({ initialActive }: OptionContractsComponentProps = {}) {
  // Form state
  const [sym, setSym] = useState('AAPL');
  const [asOf, setAsOf] = useState('');
  const [expiry, setExpiry] = useState('');
  const [rights, setRights] = useState('C,P');
  const [mode, setMode] = useState('count');
  const [expiryCount, setExpiryCount] = useState('2');
  const [spotAuto, setSpotAuto] = useState(true);
  const [spotVal, setSpotVal] = useState('');
  const [strikesPerSide, setStrikesPerSide] = useState('5');
  const [maxContracts, setMaxContracts] = useState('20');
  
  // Results state
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState('');
  const [contractsResult, setContractsResult] = useState<ContractsResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);
  const [activeTab, setActiveTab] = useState<'select' | 'enrich' | 'save' | 'results' | 'historical' | 'preview'>(initialActive || 'select');
  
  // Availability data
  const [availability, setAvailability] = useState<any>(null);
  const [expiries, setExpiries] = useState<string[]>([]);
  
  // Historical data state
  const [historicalRes, setHistoricalRes] = useState<any>(null);
  
  // Preview state
  const [previewParams, setPreviewParams] = useState({
    expiry: '',
    right: 'C',
    strike: '',
    bar_size: '1 day',
    start_date: '',
    end_date: ''
  });
  const [previewRows, setPreviewRows] = useState<any[]>([]);
  const [previewTotal, setPreviewTotal] = useState(0);
  
  // Refs for polling
  const taskIdRef = useRef<string | null>(null);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Load availability and expiries on component mount or symbol change
  useEffect(() => {
    const loadData = async () => {
      if (!sym.trim()) return;
      
      try {
        const [availRes, expiriesRes] = await Promise.all([
          api.orchestrator.optionContracts.availability(sym.trim().toUpperCase(), expiry || ''),
          api.orchestrator.optionContracts.expiries(sym.trim().toUpperCase())
        ]);
        
        setAvailability(availRes.data);
        setExpiries((expiriesRes.data as any)?.expiries || []);
      } catch (err) {
        console.error('Failed to load data:', err);
      }
    };

    loadData();
    
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
    };
  }, [sym, expiry]);

  const parseRights = (): string[] => {
    return rights.split(',').map(x => x.trim().toUpperCase()).filter(x => x === 'C' || x === 'P');
  };

  const handleSelect = async () => {
    setLoading(true);
    setStatus('Selecting...');
    setOutput('');

    try {
      const payload = {
        symbol: sym.trim().toUpperCase(),
        snapshot_date: asOf || undefined,
        expiry_filter: expiry || undefined,
        rights: parseRights(),
        strikes_per_side: Number(strikesPerSide) || undefined,
        max_contracts: Number(maxContracts) || 20,
        selection_mode: mode,
        expiry_count: Number(expiryCount) || 2,
        spot_price: spotAuto ? undefined : (Number(spotVal) || undefined),
      };

      const { data } = await api.orchestrator.optionContracts.select(payload);
      setOutput(JSON.stringify(data, null, 2));
      
      const taskId = (data as any)?.details?.task_id;
      if (!taskId) {
        setStatus('Dispatched select task (no task id returned)');
        setLoading(false);
        return;
      }

      setStatus(`Select dispatched (task ${taskId}). Waiting for result...`);
      taskIdRef.current = taskId;

      // Poll for results
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      pollTimerRef.current = setInterval(async () => {
        try {
          const { data: res } = await api.orchestrator.optionContracts.selectStatus(taskId);
          
          if ((res as any)?.contracts) {
            if (pollTimerRef.current) clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
            
            setContractsResult(res as ContractsResult);
            setOutput(JSON.stringify(res, null, 2));
            setStatus(`Selected ${(res as any).contract_count || (res as any).contracts.length} contracts for ${(res as any).symbol}`);
            setLoading(false);
            toast.success('Contract selection completed');
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      }, 2000);

    } catch (err: any) {
      setOutput(`Error: ${err.message}`);
      setStatus('Select failed');
      setLoading(false);
      toast.error(err.message || 'Failed to select contracts');
    }
  };

  const handleEnrich = async () => {
    if (!contractsResult) {
      toast.error('No contracts selected to enrich');
      return;
    }

    setLoading(true);
    setStatus('Enriching...');

    try {
      const { data } = await api.orchestrator.optionContracts.enrich({
        contracts: contractsResult.contracts
      });

      setOutput(JSON.stringify(data, null, 2));
      setStatus('Enrich completed');
      toast.success('Contract enrichment completed');
    } catch (err: any) {
      setOutput(`Error: ${err.message}`);
      setStatus('Enrich failed');
      toast.error(err.message || 'Failed to enrich contracts');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!contractsResult) {
      toast.error('No contracts to save');
      return;
    }

    setLoading(true);
    setStatus('Saving...');

    try {
      const { data } = await api.orchestrator.optionContracts.save({
        contracts: contractsResult.contracts
      });

      setOutput(JSON.stringify(data, null, 2));
      setStatus('Save completed');
      toast.success('Contracts saved successfully');
    } catch (err: any) {
      setOutput(`Error: ${err.message}`);
      setStatus('Save failed');
      toast.error(err.message || 'Failed to save contracts');
    } finally {
      setLoading(false);
    }
  };

  const handleHistorical = async () => {
    if (!contractsResult) {
      toast.error('No contracts selected for historical data fetch');
      return;
    }

    setLoading(true);
    setStatus('Fetching historical data...');

    try {
      const { data } = await api.orchestrator.optionContracts.historical({
        contracts: contractsResult.contracts,
        end_date_time: '20241201 23:59:59', // Default to recent date
        duration_str: '30 D',
        bar_size_setting: '1 day',
        timeout_seconds: 20.0
      });

      setOutput(JSON.stringify(data, null, 2));
      setHistoricalRes(data);
      setStatus('Historical data fetch completed');
      toast.success('Historical data fetch completed');
    } catch (err: any) {
      setOutput(`Error: ${err.message}`);
      setStatus('Historical data fetch failed');
      toast.error(err.message || 'Failed to fetch historical data');
    } finally {
      setLoading(false);
    }
  };

  const handleFullFlow = async () => {
    setLoading(true);
    setStatus('Running full flow...');

    try {
      // This would be a combined operation - select, enrich, save, then historical
      await handleSelect();
      // Add delay to allow select to complete
      setTimeout(async () => {
        await handleEnrich();
        setTimeout(async () => {
          await handleSave();
          setTimeout(async () => {
            await handleHistorical();
          }, 2000);
        }, 2000);
      }, 2000);
    } catch (err: any) {
      setOutput(`Error: ${err.message}`);
      setStatus('Full flow failed');
      toast.error(err.message || 'Full flow failed');
      setLoading(false);
    }
  };

  const handlePreview = async () => {
    setLoading(true);
    setStatus('Loading preview...');

    try {
      const params = {
        symbol: sym.trim().toUpperCase(),
        expiry: previewParams.expiry || undefined,
        right: previewParams.right || undefined,
        strike: previewParams.strike || undefined,
        bar_size: previewParams.bar_size || undefined,
        start_date: previewParams.start_date || undefined,
        end_date: previewParams.end_date || undefined,
        limit: 200
      };

      const { data } = await api.orchestrator.optionContracts.browseBars(params);
      setPreviewRows((data as any)?.rows || []);
      setPreviewTotal((data as any)?.total || 0);
      setStatus(`Loaded ${(data as any)?.total || 0} total records (showing up to 200)`);
      toast.success('Preview data loaded');
    } catch (err: any) {
      setStatus(`Preview failed: ${err.message}`);
      toast.error(err.message || 'Failed to load preview data');
    } finally {
      setLoading(false);
    }
  };

  const hasEnriched = !!(contractsResult && Array.isArray(contractsResult.contracts) && 
    contractsResult.contracts.some(c => c && c.conId));

  return (
    <div className="space-y-6">
      {/* Status and Availability */}
      <div className="grid gap-4 md:grid-cols-2">
        {status && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                {loading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>}
                <Badge variant={status.includes('failed') ? 'destructive' : 'secondary'}>
                  {status}
                </Badge>
              </div>
            </CardContent>
          </Card>
        )}

        {availability && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Data Availability</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm space-y-1">
                <p>Symbol: <Badge variant="outline">{availability.symbol}</Badge></p>
                <p>Available: {availability.available ? '✅ Yes' : '❌ No'}</p>
                {availability.message && <p className="text-muted-foreground">{availability.message}</p>}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      <Tabs value={activeTab} onValueChange={(value: string) => setActiveTab(value as typeof activeTab)} className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="select">Select</TabsTrigger>
          <TabsTrigger value="enrich" disabled={!contractsResult}>Enrich</TabsTrigger>
          <TabsTrigger value="save" disabled={!hasEnriched}>Save</TabsTrigger>
          <TabsTrigger value="results" disabled={!contractsResult}>Results</TabsTrigger>
          <TabsTrigger value="historical" disabled={!availability || !availability.available}>Historical</TabsTrigger>
          <TabsTrigger value="preview">Preview</TabsTrigger>
        </TabsList>

        {/* Select Tab */}
        <TabsContent value="select" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Contract Selection</CardTitle>
              <CardDescription>
                Configure parameters for option contract selection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="symbol">Symbol</Label>
                  <Input
                    id="symbol"
                    value={sym}
                    onChange={(e) => setSym(e.target.value)}
                    placeholder="AAPL"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="as-of">As-Of Date</Label>
                  <Input
                    id="as-of"
                    type="date"
                    value={asOf}
                    onChange={(e) => setAsOf(e.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="expiry-filter">Expiry Filter</Label>
                  <Select value={expiry} onValueChange={setExpiry}>
                    <SelectTrigger>
                      <SelectValue placeholder="All expiries" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All expiries</SelectItem>
                      {expiries.map((exp) => (
                        <SelectItem key={exp} value={exp}>
                          {exp}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="rights">Rights</Label>
                  <Input
                    id="rights"
                    value={rights}
                    onChange={(e) => setRights(e.target.value)}
                    placeholder="C,P"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="mode">Selection Mode</Label>
                  <Select value={mode} onValueChange={setMode}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="count">Count</SelectItem>
                      <SelectItem value="moneyness">Moneyness</SelectItem>
                      <SelectItem value="delta">Delta</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="expiry-count">Expiry Count</Label>
                  <Input
                    id="expiry-count"
                    value={expiryCount}
                    onChange={(e) => setExpiryCount(e.target.value)}
                    placeholder="2"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="strikes-per-side">Strikes Per Side</Label>
                  <Input
                    id="strikes-per-side"
                    value={strikesPerSide}
                    onChange={(e) => setStrikesPerSide(e.target.value)}
                    placeholder="5"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max-contracts">Max Contracts</Label>
                  <Input
                    id="max-contracts"
                    value={maxContracts}
                    onChange={(e) => setMaxContracts(e.target.value)}
                    placeholder="20"
                  />
                </div>
              </div>

              <Button onClick={handleSelect} disabled={loading} className="w-full">
                {loading ? 'Selecting...' : 'Select Contracts'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Enrich Tab */}
        <TabsContent value="enrich" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Enrich Contracts</CardTitle>
              <CardDescription>
                Enrich selected contracts with additional data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={handleEnrich} disabled={loading || !contractsResult}>
                {loading ? 'Enriching...' : 'Enrich Contracts'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Save Tab */}
        <TabsContent value="save" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Save Contracts</CardTitle>
              <CardDescription>
                Save enriched contracts to repository
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={handleSave} disabled={loading || !hasEnriched}>
                {loading ? 'Saving...' : 'Save Contracts'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Contract Results</CardTitle>
                <div className="flex gap-2">
                  <Button
                    variant={showTable ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setShowTable(true)}
                  >
                    Table
                  </Button>
                  <Button
                    variant={!showTable ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setShowTable(false)}
                  >
                    JSON
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {showTable && contractsResult ? (
                <ContractsTable contracts={contractsResult.contracts} />
              ) : (
                <pre className="bg-muted p-4 rounded-lg text-sm overflow-auto max-h-96 whitespace-pre-wrap">
                  {output || 'No results to display'}
                </pre>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Historical Tab */}
        <TabsContent value="historical" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Historical Data & Full Flow</CardTitle>
              <CardDescription>
                Fetch historical option bar data or run the complete workflow
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-3">
                <Button
                  onClick={handleHistorical}
                  disabled={loading || !contractsResult}
                  variant="outline"
                >
                  {loading ? 'Fetching...' : 'Run Historical'}
                </Button>
                <Button
                  onClick={handleFullFlow}
                  disabled={loading}
                >
                  {loading ? 'Running...' : 'Run Full Flow'}
                </Button>
              </div>
              
              {historicalRes && (
                <div className="flex gap-4 text-sm">
                  <span className="text-green-600">
                    Saved: {(historicalRes as any).saved || 0}
                  </span>
                  <span className="text-red-600">
                    Failed: {(historicalRes as any).failed || 0}
                  </span>
                </div>
              )}

              <pre className="bg-muted p-4 rounded-lg text-sm overflow-auto max-h-96 whitespace-pre-wrap">
                {output || 'No historical data results yet'}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Preview Tab */}
        <TabsContent value="preview" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Preview Saved Option Bars</CardTitle>
              <CardDescription>
                Browse and filter previously saved option bar data
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="preview-expiry">Expiry</Label>
                  <Select value={previewParams.expiry} onValueChange={(value) => setPreviewParams({...previewParams, expiry: value})}>
                    <SelectTrigger>
                      <SelectValue placeholder="Any expiry" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">Any expiry</SelectItem>
                      {expiries.map((exp) => (
                        <SelectItem key={exp} value={exp}>
                          {exp}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="preview-right">Right</Label>
                  <Select value={previewParams.right} onValueChange={(value) => setPreviewParams({...previewParams, right: value})}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="C">Call (C)</SelectItem>
                      <SelectItem value="P">Put (P)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="preview-strike">Strike</Label>
                  <Input
                    id="preview-strike"
                    value={previewParams.strike}
                    onChange={(e) => setPreviewParams({...previewParams, strike: e.target.value})}
                    placeholder="e.g., 135"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="preview-bar-size">Bar Size</Label>
                  <Input
                    id="preview-bar-size"
                    value={previewParams.bar_size}
                    onChange={(e) => setPreviewParams({...previewParams, bar_size: e.target.value})}
                    placeholder="1 day"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="preview-start">Start Date</Label>
                  <Input
                    id="preview-start"
                    type="date"
                    value={previewParams.start_date}
                    onChange={(e) => setPreviewParams({...previewParams, start_date: e.target.value})}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="preview-end">End Date</Label>
                  <Input
                    id="preview-end"
                    type="date"
                    value={previewParams.end_date}
                    onChange={(e) => setPreviewParams({...previewParams, end_date: e.target.value})}
                  />
                </div>
              </div>

              <div className="flex items-center gap-4">
                <Button onClick={handlePreview} disabled={loading}>
                  {loading ? 'Loading...' : 'Load Preview'}
                </Button>
                {previewTotal > 0 && (
                  <span className="text-sm text-muted-foreground">
                    {previewTotal} total (showing up to 200)
                  </span>
                )}
              </div>

              {previewRows.length > 0 && (
                <div className="overflow-auto max-h-96">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Time</TableHead>
                        <TableHead>Open</TableHead>
                        <TableHead>High</TableHead>
                        <TableHead>Low</TableHead>
                        <TableHead>Close</TableHead>
                        <TableHead>Volume</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {previewRows.map((row, index) => (
                        <TableRow key={index}>
                          <TableCell className="font-mono text-sm">{row.time}</TableCell>
                          <TableCell>{row.open}</TableCell>
                          <TableCell>{row.high}</TableCell>
                          <TableCell>{row.low}</TableCell>
                          <TableCell>{row.close}</TableCell>
                          <TableCell>{row.volume}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function ContractsTable({ contracts }: { contracts: Contract[] }) {
  if (!contracts || contracts.length === 0) {
    return <div className="text-center py-4 text-muted-foreground">No contracts to display</div>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Symbol</TableHead>
          <TableHead>Expiry</TableHead>
          <TableHead>Strike</TableHead>
          <TableHead>Right</TableHead>
          <TableHead>ConID</TableHead>
          <TableHead>Local Symbol</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {contracts.map((contract, index) => (
          <TableRow key={`${contract.symbol}-${contract.expiry}-${contract.strike}-${contract.right}-${index}`}>
            <TableCell className="font-medium">{contract.symbol}</TableCell>
            <TableCell>{contract.expiry}</TableCell>
            <TableCell>{contract.strike}</TableCell>
            <TableCell>
              <Badge variant={contract.right === 'C' ? 'default' : 'secondary'}>
                {contract.right}
              </Badge>
            </TableCell>
            <TableCell>{contract.conId || '-'}</TableCell>
            <TableCell className="font-mono text-sm">{contract.localSymbol || '-'}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}