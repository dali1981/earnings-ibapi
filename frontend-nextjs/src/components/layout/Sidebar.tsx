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
  Home,
  BarChart3,
  Sparkles,
  ChevronRight,
  Bug
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

const navigation = [
  {
    name: 'Dashboard',
    href: '/',
    icon: Home,
    isHome: true,
    gradient: 'from-blue-500 to-cyan-500',
    description: 'Overview & Stats'
  },
  {
    name: 'Calendar',
    href: '/calendar',
    icon: Calendar,
    gradient: 'from-blue-500 to-cyan-500',
    description: 'Earnings Data',
    category: 'Data Sources'
  },
  {
    name: 'Contracts',
    href: '/contracts',
    icon: Settings,
    gradient: 'from-gray-500 to-slate-600',
    description: 'Contract Setup',
    category: 'Management'
  },
  {
    name: 'Equity History',
    href: '/equity',
    icon: TrendingUp,
    gradient: 'from-green-500 to-emerald-500',
    description: 'Historical Bars',
    category: 'Analytics'
  },
  {
    name: 'Option Chains',
    href: '/chains',
    icon: Database,
    gradient: 'from-purple-500 to-violet-500',
    description: 'Chain Snapshots',
    category: 'Data Sources'
  },
  {
    name: 'Option Workflow',
    href: '/options',
    icon: Building2,
    gradient: 'from-orange-500 to-red-500',
    description: 'Contract Flow',
    category: 'Workflow'
  },
  {
    name: 'Option History',
    href: '/history',
    icon: History,
    gradient: 'from-indigo-500 to-blue-500',
    description: 'Historical Data',
    category: 'Analytics'
  },
  {
    name: 'Sentry Test',
    href: '/sentry-test',
    icon: Bug,
    gradient: 'from-yellow-500 to-orange-500',
    description: 'Test Observability',
    category: 'Development',
    badge: process.env.NODE_ENV === 'development' ? 'DEV' : null
  },
];

const categories = ['Data Sources', 'Analytics', 'Management', 'Workflow', 'Development'];

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();

  return (
    <>
      {/* Mobile menu button */}
      <Button
        variant="ghost"
        size="sm"
        className="fixed top-4 left-4 z-50 md:hidden bg-white dark:bg-gray-800 shadow-lg"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        <span className="sr-only">Toggle sidebar</span>
      </Button>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          'fixed left-0 top-0 z-50 h-full w-72 transform bg-gradient-to-b from-white to-gray-50 dark:from-gray-900 dark:to-gray-950 border-r border-border/50 shadow-xl transition-transform duration-300 ease-in-out md:relative md:translate-x-0',
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        )}
      >
        <div className="flex h-full flex-col">
          {/* Enhanced Header */}
          <div className="relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800"></div>
            <div className="relative p-6 text-white">
              <Link
                href="/"
                className="flex items-center gap-3 hover:scale-105 transition-transform duration-200"
                onClick={() => setIsOpen(false)}
              >
                <div className="p-2 bg-white/20 backdrop-blur-sm rounded-xl">
                  <BarChart3 className="h-6 w-6" />
                </div>
                <div>
                  <h1 className="text-lg font-bold">Trading Platform</h1>
                  <p className="text-blue-100 text-xs">Command Center</p>
                </div>
              </Link>

              {/* Live indicator */}
              <div className="flex items-center gap-2 mt-4 text-blue-100">
                <Sparkles className="h-3 w-3 animate-pulse" />
                <span className="text-xs font-medium">Live Market Data</span>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 overflow-y-auto">
            <div className="space-y-6">
              {/* Dashboard */}
              <div>
                {navigation.filter(item => item.isHome).map((item) => {
                  const isActive = pathname === item.href;
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={cn(
                        'group flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200',
                        isActive
                          ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                          : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      )}
                      onClick={() => setIsOpen(false)}
                    >
                      <div className={cn(
                        "p-2 rounded-lg transition-all duration-200",
                        isActive
                          ? "bg-white/20"
                          : "bg-gray-100 dark:bg-gray-800 group-hover:scale-110"
                      )}>
                        <item.icon className="h-4 w-4" />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold">{item.name}</div>
                        <div className="text-xs opacity-75">{item.description}</div>
                      </div>
                      {isActive && <ChevronRight className="h-4 w-4" />}
                    </Link>
                  );
                })}
              </div>

              {/* Categorized Services */}
              {categories.map((category) => (
                <div key={category}>
                  <div className="flex items-center gap-2 px-4 mb-3">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      {category}
                    </h3>
                    <Badge variant="secondary" className="text-xs">
                      {navigation.filter(item => item.category === category).length}
                    </Badge>
                  </div>
                  <div className="space-y-2">
                    {navigation.filter(item => item.category === category).map((item) => {
                      const isActive = pathname === item.href;
                      return (
                        <Link
                          key={item.name}
                          href={item.href}
                          className={cn(
                            'group flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200 relative overflow-hidden',
                            isActive
                              ? 'bg-primary text-primary-foreground shadow-md'
                              : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                          )}
                          onClick={() => setIsOpen(false)}
                        >
                          {/* Gradient background on hover */}
                          <div className={cn(
                            "absolute inset-0 bg-gradient-to-br opacity-0 transition-opacity duration-300",
                            item.gradient,
                            !isActive && "group-hover:opacity-5"
                          )}></div>

                          <div className={cn(
                            "relative p-2 rounded-lg transition-all duration-200",
                            isActive
                              ? "bg-white/20"
                              : "bg-gray-100 dark:bg-gray-800 group-hover:scale-110"
                          )}>
                            <item.icon className="h-4 w-4" />
                          </div>

                          <div className="relative flex-1">
                            <div className="font-semibold">{item.name}</div>
                            <div className="text-xs opacity-75">{item.description}</div>
                          </div>

                          {isActive && <ChevronRight className="h-4 w-4 relative" />}
                        </Link>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </nav>

          {/* Enhanced Footer */}
          <div className="p-4 border-t border-border bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs font-medium text-green-600">All Systems Online</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Trading Microservices Platform
            </p>
            <p className="text-xs text-muted-foreground">
              Built with Next.js & shadcn/ui
            </p>
          </div>
        </div>
      </div>
    </>
  );
}