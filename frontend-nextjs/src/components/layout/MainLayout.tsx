'use client';

import { Sidebar } from './Sidebar';
import { Breadcrumb } from '../ui/breadcrumb';

interface MainLayoutProps {
  children: React.ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="flex h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-950 dark:to-gray-900">
      <Sidebar />

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="container mx-auto p-6 md:p-8 max-w-7xl">
          <div className="mb-6">
            <Breadcrumb />
          </div>
          <div className="min-h-full">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
}