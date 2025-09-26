import { MainLayout } from '@/components/layout/MainLayout';
import OrchestratorComponent from '@/components/trading/Orchestrator/Orchestrator';

export default function ContractsPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="space-y-4">
          <h1 className="text-3xl font-bold tracking-tight">
            Contract Management
          </h1>
          <p className="text-muted-foreground">
            Search, populate, and setup contract descriptions through the orchestrator service.
          </p>
        </div>
        
        <OrchestratorComponent />
      </div>
    </MainLayout>
  );
}