import { MainLayout } from '@/components/layout/MainLayout';
import OptionHistoryComponent from '@/components/trading/History/OptionHistory';

export default function HistoryPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="space-y-4">
          <h1 className="text-3xl font-bold tracking-tight">
            Option Historical Data
          </h1>
          <p className="text-muted-foreground">
            Fetch and analyze historical option bar data with full workflow support.
          </p>
        </div>
        
        <OptionHistoryComponent />
      </div>
    </MainLayout>
  );
}