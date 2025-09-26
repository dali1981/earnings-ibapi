import { MainLayout } from '@/components/layout/MainLayout';
import CalendarComponent from '@/components/trading/Calendar/Calendar';
import { Calendar, TrendingUp, Clock } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export default function CalendarPage() {
  return (
    <MainLayout>
      <div className="space-y-8">
        {/* Enhanced Page Header */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-500 via-cyan-500 to-blue-600 p-8 text-white">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-transparent"></div>
          <div className="relative">
            <div className="flex items-center justify-between">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-white/20 backdrop-blur-sm rounded-xl">
                    <Calendar className="h-8 w-8" />
                  </div>
                  <div>
                    <h1 className="text-4xl font-bold tracking-tight">
                      Earnings Calendar
                    </h1>
                    <p className="text-blue-100 text-lg">
                      Track upcoming earnings announcements and market events
                    </p>
                  </div>
                </div>
              </div>
              <div className="flex flex-col gap-3 text-right">
                <Badge className="bg-white/20 text-white border-white/30 hover:bg-white/30">
                  <TrendingUp className="h-3 w-3 mr-1" />
                  Live Data
                </Badge>
                <Badge className="bg-white/20 text-white border-white/30 hover:bg-white/30">
                  <Clock className="h-3 w-3 mr-1" />
                  Real-time
                </Badge>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <CalendarComponent />
      </div>
    </MainLayout>
  );
}