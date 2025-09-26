import { MainLayout } from '@/components/layout/MainLayout';
import ChainsComponent from '@/components/trading/Chains/Chains';

export default function ChainsPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="space-y-4">
          <h1 className="text-3xl font-bold tracking-tight">
            Option Chain Snapshots  
          </h1>
          <p className="text-muted-foreground">
            Browse option chain snapshots and data.
          </p>
        </div>
        
        <ChainsComponent />
      </div>
    </MainLayout>
  );
}