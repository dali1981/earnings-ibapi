import { MainLayout } from '@/components/layout/MainLayout';
import OptionContractsComponent from '@/components/trading/Contracts/OptionContracts';

export default function OptionsPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="space-y-4">
          <h1 className="text-3xl font-bold tracking-tight">
            Option Contract Workflow
          </h1>
          <p className="text-muted-foreground">
            Select, enrich, and save option contracts with advanced filtering and monitoring capabilities.
          </p>
        </div>
        
        <OptionContractsComponent />
      </div>
    </MainLayout>
  );
}