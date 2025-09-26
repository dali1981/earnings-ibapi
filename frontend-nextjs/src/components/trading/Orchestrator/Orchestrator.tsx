'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from 'sonner';
import { api } from '@/lib/api';
import type { ContractSearchRequest } from '@/types/api';

function parseList(s: string): string[] {
  return (s || '').split(',').map(x => x.trim()).filter(Boolean);
}

export default function OrchestratorComponent() {
  const [symbols, setSymbols] = useState('AAPL,MSFT');
  const [secTypes, setSecTypes] = useState('STK');
  const [currency, setCurrency] = useState('USD');
  const [exchanges, setExchanges] = useState('NYSE,NASDAQ,AMEX,ARCA,BATS,IEX');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastAction, setLastAction] = useState('');

  const handleRun = async (kind: 'search' | 'details' | 'setup') => {
    setLoading(true);
    setLastAction(kind);

    try {
      const payload: ContractSearchRequest = {
        symbols: parseList(symbols).map(s => s.toUpperCase()),
        sec_types: parseList(secTypes).map(s => s.toUpperCase()),
        currency: (currency || 'USD').toUpperCase(),
        exchanges: parseList(exchanges).map(s => s.toUpperCase()),
      };

      let response;
      let actionLabel;

      switch (kind) {
        case 'search':
          response = await api.orchestrator.searchContracts(payload);
          actionLabel = 'Search Descriptions';
          break;
        case 'details':
          response = await api.orchestrator.getContractDetails({ symbols: payload.symbols });
          actionLabel = 'Fetch Details';
          break;
        case 'setup':
          response = await api.orchestrator.setupContracts(payload);
          actionLabel = 'Setup (Search + Details)';
          break;
      }

      setOutput(JSON.stringify(response.data, null, 2));
      toast.success(`${actionLabel} completed successfully`);
    } catch (err: any) {
      const errorMsg = `Error: ${err.message}`;
      setOutput(errorMsg);
      toast.error(err.message || `Failed to ${kind} contracts`);
    } finally {
      setLoading(false);
    }
  };

  const getActionVariant = (action: string) => {
    if (loading && lastAction === action) return 'default';
    return 'outline';
  };

  return (
    <div className="space-y-6">
      {/* Configuration Card */}
      <Card>
        <CardHeader>
          <CardTitle>Contract Search Configuration</CardTitle>
          <CardDescription>
            Configure parameters for contract search and setup operations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="symbols">Symbols</Label>
                <Input
                  id="symbols"
                  value={symbols}
                  onChange={(e) => setSymbols(e.target.value)}
                  placeholder="AAPL, MSFT, TSLA"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="sec-types">Security Types</Label>
                <Input
                  id="sec-types"
                  value={secTypes}
                  onChange={(e) => setSecTypes(e.target.value)}
                  placeholder="STK"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="currency">Currency</Label>
                <Input
                  id="currency"
                  value={currency}
                  onChange={(e) => setCurrency(e.target.value)}
                  placeholder="USD"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="exchanges">Exchanges</Label>
                <Input
                  id="exchanges"
                  value={exchanges}
                  onChange={(e) => setExchanges(e.target.value)}
                  placeholder="NYSE,NASDAQ,AMEX,ARCA,BATS,IEX"
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Actions Card */}
      <Card>
        <CardHeader>
          <CardTitle>Actions</CardTitle>
          <CardDescription>
            Run contract operations with the configured parameters
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Button
              variant={getActionVariant('search')}
              onClick={() => handleRun('search')}
              disabled={loading}
            >
              {loading && lastAction === 'search' ? 'Searching...' : 'Search Descriptions'}
            </Button>
            
            <Button
              variant={getActionVariant('details')}
              onClick={() => handleRun('details')}
              disabled={loading}
            >
              {loading && lastAction === 'details' ? 'Fetching...' : 'Fetch Details'}
            </Button>
            
            <Button
              variant={getActionVariant('setup')}
              onClick={() => handleRun('setup')}
              disabled={loading}
            >
              {loading && lastAction === 'setup' ? 'Setting up...' : 'Setup (Search + Details)'}
            </Button>
          </div>

          {lastAction && (
            <div className="mt-4">
              <Badge variant="secondary" className="text-sm">
                Last action: {lastAction}
              </Badge>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results Card */}
      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>
            Output from orchestrator operations
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          ) : (
            <pre className="bg-muted p-4 rounded-lg text-sm overflow-auto max-h-96 whitespace-pre-wrap">
              {output || 'No results yet - run an action above to see output'}
            </pre>
          )}
        </CardContent>
      </Card>

      {/* Help Card */}
      <Card>
        <CardHeader>
          <CardTitle>Operation Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="text-sm space-y-1">
            <p><strong>Search Descriptions:</strong> Find and save contract descriptions for the specified symbols</p>
            <p><strong>Fetch Details:</strong> Get detailed contract information for symbols (requires existing descriptions)</p>
            <p><strong>Setup:</strong> Run both search and fetch operations in sequence for complete contract setup</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}