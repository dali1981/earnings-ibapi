'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from 'sonner';
import { api } from '@/lib/api';

export default function EquityComponent() {
  // Repository read form state
  const [sym, setSym] = useState('AAPL');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [bar, setBar] = useState('1 day');
  
  // TWS fetch form state
  const [symbols, setSymbols] = useState('AAPL');
  const [duration, setDuration] = useState('5 D');
  const [barSize, setBarSize] = useState('1 day');
  const [endDate, setEndDate] = useState('');
  
  // Output state
  const [status, setStatus] = useState('');
  const [output, setOutput] = useState('');
  const [datesInfo, setDatesInfo] = useState('');
  const [loading, setLoading] = useState(false);

  const durationOptions = [
    { value: '1 D', label: '1 Day' },
    { value: '5 D', label: '5 Days' },
    { value: '1 W', label: '1 Week' },
    { value: '2 W', label: '2 Weeks' },
    { value: '1 M', label: '1 Month' },
    { value: '3 M', label: '3 Months' },
    { value: '6 M', label: '6 Months' },
    { value: '1 Y', label: '1 Year' },
  ];

  const barSizeOptions = [
    { value: '1 min', label: '1 minute' },
    { value: '5 mins', label: '5 minutes' },
    { value: '15 mins', label: '15 minutes' },
    { value: '30 mins', label: '30 minutes' },
    { value: '1 hour', label: '1 hour' },
    { value: '1 day', label: '1 day' },
  ];

  const handleReadRepo = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setStatus('Reading from repository...');
    setOutput('Loading...');
    
    try {
      const { data } = await api.equity.getBars(sym, start || undefined, end || undefined, bar);
      setOutput(JSON.stringify(data, null, 2));
      setStatus(`Read ${data.rows || 0} bars from repository`);
      toast.success(`Successfully read ${data.rows || 0} bars`);
    } catch (err: any) {
      const errorMsg = `Error: ${err.message}`;
      setOutput(errorMsg);
      setStatus('Read failed');
      toast.error(err.message || 'Failed to read from repository');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadDates = async () => {
    setDatesInfo('Loading available dates...');
    
    try {
      const { data } = await api.equity.getDates(sym, bar);
      const dates = data.dates || [];
      
      if (dates.length > 0) {
        const first = dates[0];
        const last = dates[dates.length - 1];
        setDatesInfo(`Available: ${dates.length} dates. Range: ${first} to ${last}`);
      } else {
        setDatesInfo('No dates available');
      }
    } catch (err: any) {
      setDatesInfo(`Error: ${err.message}`);
      toast.error(err.message || 'Failed to load dates');
    }
  };

  const handleFetchTWS = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!symbols.trim()) {
      toast.error('Please enter symbol(s)');
      return;
    }

    setLoading(true);
    const endDateTime = endDate 
      ? `${endDate.replace(/-/g, '')} 23:59:59`
      : new Date().toISOString().slice(0, 10).replace(/-/g, '') + ' 23:59:59';

    setStatus('Fetching from TWS...');
    setOutput('Dispatching request to orchestrator...');

    try {
      const { data } = await api.orchestrator.fetchHistoricalBars({
        symbols: symbols.split(',').map(s => s.trim()),
        end_date_time: endDateTime,
        duration_str: duration,
        bar_size_setting: barSize,
        repository_type: 'equity',
      });

      setOutput(JSON.stringify(data, null, 2));
      setStatus(`Fetching ${data.processed || 0} symbol(s) from TWS. Job: ${data.details?.group_id || data.details?.chain_id || 'N/A'}`);
      toast.success(`Successfully dispatched request for ${data.processed || 0} symbols`);
    } catch (err: any) {
      const errorMsg = `Error: ${err.message}`;
      setOutput(errorMsg);
      setStatus('Fetch failed');
      toast.error(err.message || 'Failed to fetch from TWS');
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
            <Badge variant={status.includes('failed') ? 'destructive' : 'secondary'}>
              {status}
            </Badge>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="read" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="read">Read Repository</TabsTrigger>
          <TabsTrigger value="fetch">Fetch from TWS</TabsTrigger>
        </TabsList>

        {/* Read from Repository Tab */}
        <TabsContent value="read" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Read from Repository</CardTitle>
              <CardDescription>
                Read historical equity bars from the local repository
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleReadRepo} className="grid gap-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="repo-symbol">Symbol</Label>
                    <Input
                      id="repo-symbol"
                      value={sym}
                      onChange={(e) => setSym(e.target.value)}
                      placeholder="AAPL"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="repo-bar-size">Bar Size</Label>
                    <Input
                      id="repo-bar-size"
                      value={bar}
                      onChange={(e) => setBar(e.target.value)}
                      placeholder="1 day"
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="repo-start">Start Date</Label>
                    <Input
                      id="repo-start"
                      type="date"
                      value={start}
                      onChange={(e) => setStart(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="repo-end">End Date</Label>
                    <Input
                      id="repo-end"
                      type="date"
                      value={end}
                      onChange={(e) => setEnd(e.target.value)}
                    />
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button type="submit" disabled={loading}>
                    Read from Repository
                  </Button>
                  <Button type="button" variant="outline" onClick={handleLoadDates}>
                    Available Dates
                  </Button>
                </div>
              </form>

              {datesInfo && (
                <div className="text-sm text-muted-foreground p-3 bg-muted rounded-lg">
                  {datesInfo}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Fetch from TWS Tab */}
        <TabsContent value="fetch" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Fetch from TWS</CardTitle>
              <CardDescription>
                Fetch new historical equity data from TWS via orchestrator
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleFetchTWS} className="grid gap-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="tws-symbols">Symbol(s)</Label>
                    <Input
                      id="tws-symbols"
                      value={symbols}
                      onChange={(e) => setSymbols(e.target.value)}
                      placeholder="AAPL or AAPL,MSFT"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="tws-end-date">End Date</Label>
                    <Input
                      id="tws-end-date"
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="tws-duration">Duration</Label>
                    <Select value={duration} onValueChange={setDuration}>
                      <SelectTrigger id="tws-duration">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {durationOptions.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="tws-bar-size">Bar Size</Label>
                    <Select value={barSize} onValueChange={setBarSize}>
                      <SelectTrigger id="tws-bar-size">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {barSizeOptions.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Button type="submit" disabled={loading}>
                  Fetch from TWS (via Orchestrator)
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

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