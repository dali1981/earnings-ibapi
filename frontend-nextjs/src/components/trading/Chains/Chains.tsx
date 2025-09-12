'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from 'sonner';
import { api } from '@/lib/api';

export default function ChainsComponent() {
  const [sym, setSym] = useState('AAPL');
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState('');
  const [snapDate, setSnapDate] = useState('');
  const [asOf, setAsOf] = useState('');
  const [asOfList, setAsOfList] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const handleLatestSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setStatus('Loading...');
    setOutput('');

    try {
      const { data } = await api.chains.getLatest(sym);
      setOutput(JSON.stringify(data, null, 2));
      setStatus('OK');
      toast.success('Successfully loaded latest option chain');
    } catch (err: any) {
      const msg = err?.message || String(err);
      setOutput(`Error: ${msg}`);
      setStatus('Failed');
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadAsOfs = async () => {
    setLoading(true);
    setStatus('Loading as-of list...');
    setOutput('');

    try {
      const { data } = await api.chains.getAsOfs(sym);
      const asOfData = Array.isArray(data) ? data : (data as any)?.as_of_list || [];
      setAsOfList(asOfData);
      
      if ((data as any)?.latest) {
        setAsOf((data as any).latest);
      }
      
      setOutput(JSON.stringify(data, null, 2));
      setStatus('OK');
      toast.success(`Loaded ${asOfData.length} as-of dates`);
    } catch (err: any) {
      const msg = err?.message || String(err);
      setOutput(`Error: ${msg}`);
      setStatus('Failed');
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleAsOfAction = async (action: 'summary' | 'expiries' | 'strikes' | 'params') => {
    if (!asOf) {
      toast.error('Please pick an as-of date');
      return;
    }

    setLoading(true);
    setStatus(`Loading ${action}...`);
    setOutput('');

    try {
      let data;
      switch (action) {
        case 'summary':
          ({ data } = await api.chains.getSummary(sym, asOf));
          break;
        case 'expiries':
          ({ data } = await api.chains.getExpiries(sym, asOf));
          break;
        case 'strikes':
          ({ data } = await api.chains.getStrikes(sym, asOf));
          break;
        case 'params':
          ({ data } = await api.chains.getParams(sym, asOf));
          break;
      }

      setOutput(JSON.stringify(data, null, 2));
      setStatus('OK');
      toast.success(`Successfully loaded ${action}`);
    } catch (err: any) {
      const msg = err?.message || String(err);
      setOutput(`Error: ${msg}`);
      setStatus('Failed');
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleFetchViaOrchestrator = async (e: React.FormEvent) => {
    e.preventDefault();
    const s = sym.trim();
    
    if (!s) {
      toast.error('Please enter a symbol');
      return;
    }

    setLoading(true);
    setStatus('Dispatching snapshot fetch to orchestrator...');
    setOutput('');

    try {
      const payload: any = { symbols: [s.toUpperCase()] };
      if (snapDate) {
        payload.snapshot_date = snapDate;
      }

      const { data } = await api.orchestrator.fetchHistoricalBars({
        symbols: [s.toUpperCase()],
        end_date_time: new Date().toISOString().slice(0, 10).replace(/-/g, '') + ' 23:59:59',
        duration_str: '1 D',
        bar_size_setting: '1 day',
        repository_type: 'option',
        ...payload
      });

      setOutput(JSON.stringify(data, null, 2));
      setStatus(`Dispatched ${(data as any)?.details?.dispatched ?? 1} task(s). Job ID: ${(data as any)?.details?.group_id || (data as any)?.job_id || 'N/A'}`);
      toast.success('Successfully dispatched option chain fetch');
    } catch (err: any) {
      const msg = err?.message || String(err);
      setOutput(`Error: ${msg}`);
      setStatus('Dispatch failed');
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Status Badge */}
      {status && (
        <Card>
          <CardContent className="pt-6">
            <Badge variant={status.includes('Failed') ? 'destructive' : 'secondary'}>
              {status}
            </Badge>
          </CardContent>
        </Card>
      )}

      {/* Latest Chain Card */}
      <Card>
        <CardHeader>
          <CardTitle>Latest Option Chain</CardTitle>
          <CardDescription>
            Read latest option chain data from repository
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form onSubmit={handleLatestSubmit} className="flex gap-4 items-end">
            <div className="space-y-2 flex-1">
              <Label htmlFor="latest-symbol">Symbol</Label>
              <Input
                id="latest-symbol"
                value={sym}
                onChange={(e) => setSym(e.target.value)}
                placeholder="AAPL"
              />
            </div>
            <Button type="submit" disabled={loading}>
              Read Latest (repo)
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Historical Data Card */}
      <Card>
        <CardHeader>
          <CardTitle>Historical Chain Data</CardTitle>
          <CardDescription>
            Browse historical option chain snapshots
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2 items-center flex-wrap">
            <Button variant="outline" onClick={handleLoadAsOfs} disabled={loading}>
              List As-Of Dates
            </Button>
            
            <div className="flex items-center gap-2">
              <Label htmlFor="as-of-select">As-Of Date:</Label>
              <Select value={asOf} onValueChange={setAsOf}>
                <SelectTrigger id="as-of-select" className="w-48">
                  <SelectValue placeholder="Select date" />
                </SelectTrigger>
                <SelectContent>
                  {asOfList.map((date) => (
                    <SelectItem key={date} value={date}>
                      {date}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex gap-2 flex-wrap">
            <Button 
              variant="outline" 
              onClick={() => handleAsOfAction('summary')}
              disabled={loading || !asOf}
            >
              Summary
            </Button>
            <Button 
              variant="outline" 
              onClick={() => handleAsOfAction('expiries')}
              disabled={loading || !asOf}
            >
              Expiries
            </Button>
            <Button 
              variant="outline" 
              onClick={() => handleAsOfAction('strikes')}
              disabled={loading || !asOf}
            >
              Strikes
            </Button>
            <Button 
              variant="outline" 
              onClick={() => handleAsOfAction('params')}
              disabled={loading || !asOf}
            >
              Params
            </Button>
          </div>
        </CardContent>
      </Card>

      <Separator />

      {/* Fetch via Orchestrator Card */}
      <Card>
        <CardHeader>
          <CardTitle>Fetch New Data</CardTitle>
          <CardDescription>
            Fetch new option chain snapshots via orchestrator
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form onSubmit={handleFetchViaOrchestrator} className="grid gap-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="fetch-symbol">Symbol</Label>
                <Input
                  id="fetch-symbol"
                  value={sym}
                  onChange={(e) => setSym(e.target.value)}
                  placeholder="AAPL"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="snap-date">Snapshot Date (optional)</Label>
                <Input
                  id="snap-date"
                  type="date"
                  value={snapDate}
                  onChange={(e) => setSnapDate(e.target.value)}
                />
              </div>
            </div>
            
            <Button type="submit" disabled={loading}>
              Fetch via Orchestrator
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Results */}
      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : (
            <pre className="bg-muted p-4 rounded-lg text-sm overflow-auto max-h-96 whitespace-pre-wrap">
              {output || 'No data to display'}
            </pre>
          )}
        </CardContent>
      </Card>
    </div>
  );
}