import { MainLayout } from '@/components/layout/MainLayout';
import CalendarComponent from '@/components/trading/Calendar/Calendar';

export default function CalendarPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="space-y-4">
          <h1 className="text-3xl font-bold tracking-tight">
            Earnings Calendar
          </h1>
          <p className="text-muted-foreground">
            View and fetch upcoming earnings announcements from multiple sources.
          </p>
        </div>
        
        <CalendarComponent />
      </div>
    </MainLayout>
  );
}