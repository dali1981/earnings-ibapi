'use client';

import { Sidebar } from './Sidebar';
import { Breadcrumb } from '../ui/breadcrumb';

interface MainLayoutProps {
  children: React.ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      
      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="container mx-auto p-6 md:p-8">
          <Breadcrumb />
          {children}
        </div>
      </main>
    </div>
  );
}