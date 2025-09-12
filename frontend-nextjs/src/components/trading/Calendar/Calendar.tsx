'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from 'sonner';
import { api } from '@/lib/api';
import type { CalendarEvent } from '@/types/api';

export default function CalendarComponent() {
  // Form state
  const [getDays, setGetDays] = useState(7);
  const [getSymbols, setGetSymbols] = useState('');
  const [fetchDays, setFetchDays] = useState(7);
  const [fetchSource, setFetchSource] = useState('nasdaq');
  const [fetchSymbols, setFetchSymbols] = useState('');
  const [viewMode, setViewMode] = useState<'table' | 'json'>('table');
  
  // Metadata state
  const [metadata, setMetadata] = useState({ source: '-', collection: '-' });

  // Query for getting data from repository
  const { 
    data: calendarData, 
    isLoading: isGetting, 
    refetch: refetchGet 
  } = useQuery({
    queryKey: ['calendar', 'get', getDays, getSymbols],
    queryFn: async () => {
      const response = await api.calendar.getUpcoming(getDays, getSymbols || undefined);
      // Extract metadata from response headers
      const source = response.response.headers.get('X-Calendar-Source') || '-';
      const collection = response.response.headers.get('X-Calendar-Collection-Date') || '-';
      setMetadata({ source, collection });
      return response.data;
    },
    enabled: false, // Only run when explicitly triggered
  });

  // Mutation for fetching from external sources
  const [isFetching, setIsFetching] = useState(false);
  const handleFetch = async () => {
    if (fetchSource === 'yahoo' && !fetchSymbols) {
      toast.error('Yahoo requires symbols (comma-separated)');
      return;
    }

    setIsFetching(true);
    try {
      const response = await api.calendar.fetchUpcoming(
        fetchDays, 
        fetchSource, 
        fetchSymbols || undefined
      );
      
      // Extract metadata
      const source = response.response.headers.get('X-Calendar-Source') || '-';
      const collection = response.response.headers.get('X-Calendar-Collection-Date') || '-';
      setMetadata({ source, collection });
      
      toast.success('Calendar data fetched successfully');
      // Refetch the get query to show updated data
      refetchGet();
    } catch (error: any) {
      toast.error(error.message || 'Failed to fetch calendar data');
    } finally {
      setIsFetching(false);
    }
  };

  const handleGet = () => {
    refetchGet();
  };

  const events = calendarData?.events || [];

  return (
    <div className="space-y-6">
      {/* Metadata */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Data Source Information</CardTitle>
            <div className="flex gap-2">
              <Badge variant="outline">Last collection: {metadata.collection}</Badge>
              <Badge variant="outline">Source: {metadata.source}</Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Actions */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Get from Repository */}
        <Card>
          <CardHeader>
            <CardTitle>Read from Repository</CardTitle>
            <CardDescription>
              Get earnings data from the local repository
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4">
              <div className="space-y-2">
                <Label htmlFor="get-days">Days Ahead</Label>
                <Input
                  id="get-days"
                  type="number"
                  min={0}
                  max={180}
                  value={getDays}
                  onChange={(e) => setGetDays(Number(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="get-symbols">Symbols (optional)</Label>
                <Input
                  id="get-symbols"
                  placeholder="AAPL,MSFT"
                  value={getSymbols}
                  onChange={(e) => setGetSymbols(e.target.value)}
                />
              </div>
              <Button onClick={handleGet} disabled={isGetting}>
                {isGetting ? 'Loading...' : 'Get (repo only)'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Fetch from External */}
        <Card>
          <CardHeader>
            <CardTitle>Fetch from External</CardTitle>
            <CardDescription>
              Fetch earnings data from external sources and persist
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4">
              <div className="space-y-2">
                <Label htmlFor="fetch-days">Days Ahead</Label>
                <Input
                  id="fetch-days"
                  type="number"
                  min={0}
                  max={180}
                  value={fetchDays}
                  onChange={(e) => setFetchDays(Number(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="fetch-source">Source</Label>
                <Select value={fetchSource} onValueChange={setFetchSource}>
                  <SelectTrigger id="fetch-source">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="nasdaq">NASDAQ</SelectItem>
                    <SelectItem value="yahoo">Yahoo</SelectItem>
                    <SelectItem value="all">All</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="fetch-symbols">
                  Symbols {fetchSource === 'yahoo' && '(required for Yahoo)'}
                </Label>
                <Input
                  id="fetch-symbols"
                  placeholder="AAPL,MSFT"
                  value={fetchSymbols}
                  onChange={(e) => setFetchSymbols(e.target.value)}
                />
              </div>
              <Button onClick={handleFetch} disabled={isFetching}>
                {isFetching ? 'Fetching...' : 'Fetch & persist (external)'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Results */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Results</CardTitle>
            <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'table' | 'json')}>
              <TabsList>
                <TabsTrigger value="table">Table</TabsTrigger>
                <TabsTrigger value="json">JSON</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </CardHeader>
        <CardContent>
          {isGetting ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : (
            <Tabs value={viewMode}>
              <TabsContent value="table">
                <CalendarTable events={events} />
              </TabsContent>
              <TabsContent value="json">
                <pre className="bg-muted p-4 rounded-lg text-sm overflow-auto max-h-96">
                  {JSON.stringify(events, null, 2)}
                </pre>
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function CalendarTable({ events }: { events: CalendarEvent[] }) {
  if (!events || events.length === 0) {
    return (
      <div className="text-center py-6 text-muted-foreground">
        No earnings events found
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Symbol</TableHead>
          <TableHead>Company Name</TableHead>
          <TableHead>Earnings Date</TableHead>
          <TableHead>Time</TableHead>
          <TableHead>EPS Estimate</TableHead>
          <TableHead>Revenue Estimate</TableHead>
          <TableHead>Source</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {events.map((event, index) => (
          <TableRow key={`${event.symbol}-${event.earnings_date}-${index}`}>
            <TableCell className="font-medium">{event.symbol}</TableCell>
            <TableCell>{event.company_name || '-'}</TableCell>
            <TableCell>{event.earnings_date}</TableCell>
            <TableCell>{event.time || '-'}</TableCell>
            <TableCell>{event.eps_estimate || '-'}</TableCell>
            <TableCell>{event.revenue_estimate || '-'}</TableCell>
            <TableCell>
              <Badge variant="secondary">{event.source}</Badge>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}