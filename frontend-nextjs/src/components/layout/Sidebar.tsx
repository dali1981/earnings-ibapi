'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { 
  Calendar,
  TrendingUp, 
  Database,
  Building2,
  History,
  Settings,
  Menu,
  X,
  Home
} from 'lucide-react';
import { Button } from '@/components/ui/button';

const navigation = [
  { name: 'Home', href: '/', icon: Home, isHome: true },
  { name: 'Calendar', href: '/calendar', icon: Calendar },
  { name: 'Contracts', href: '/contracts', icon: Settings },
  { name: 'Equity History', href: '/equity', icon: TrendingUp },
  { name: 'Option Chains', href: '/chains', icon: Database },
  { name: 'Option Workflow', href: '/options', icon: Building2 },
  { name: 'Option History', href: '/history', icon: History },
];

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();

  return (
    <>
      {/* Mobile menu button */}
      <Button
        variant="ghost"
        size="sm"
        className="fixed top-4 left-4 z-50 md:hidden"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        <span className="sr-only">Toggle sidebar</span>
      </Button>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          'fixed left-0 top-0 z-50 h-full w-64 transform bg-card border-r border-border transition-transform duration-200 ease-in-out md:relative md:translate-x-0',
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        )}
      >
        <div className="flex h-full flex-col">
          {/* Header */}
          <div className="flex h-16 items-center justify-between px-4 border-b border-border">
            <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
              <Home className="h-5 w-5 text-primary" />
              <h1 className="text-lg font-semibold text-foreground">
                Trading Services
              </h1>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-3 py-4">
            <ul className="space-y-2">
              {navigation.map((item) => {
                const isActive = pathname === item.href;
                const isHome = item.isHome;
                return (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className={cn(
                        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                        isActive
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
                        isHome && 'border border-border/50'
                      )}
                      onClick={() => setIsOpen(false)}
                    >
                      <item.icon className={cn("h-4 w-4", isHome && "text-primary")} />
                      {item.name}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-border">
            <p className="text-xs text-muted-foreground">
              Trading Microservices
            </p>
            <p className="text-xs text-muted-foreground">
              Built with Next.js
            </p>
          </div>
        </div>
      </div>
    </>
  );
}