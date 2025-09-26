import { MainLayout } from '@/components/layout/MainLayout';
import EquityComponent from '@/components/trading/Equity/Equity';

export default function EquityPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="space-y-4">
          <h1 className="text-3xl font-bold tracking-tight">
            Equity Historical Data
          </h1>
          <p className="text-muted-foreground">
            Access historical equity bar data from repository and fetch new data from TWS.
          </p>
        </div>
        
        <EquityComponent />
      </div>
    </MainLayout>
  );
}