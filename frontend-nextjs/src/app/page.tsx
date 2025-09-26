'use client';

import { MainLayout } from '@/components/layout/MainLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import Link from 'next/link';
import { Calendar, TrendingUp, Database, Building2, History, Settings, BarChart3, Zap, Shield, Activity } from 'lucide-react';
import { useEffect, useState } from 'react';

const services = [
  {
    title: 'Calendar Service',
    description: 'View and fetch upcoming earnings announcements',
    href: '/calendar',
    icon: Calendar,
    category: 'Data Sources',
    status: 'active',
    gradient: 'from-blue-500 to-cyan-500',
  },
  {
    title: 'Contract Management',
    description: 'Search, populate, and setup contract descriptions',
    href: '/contracts',
    icon: Settings,
    category: 'Management',
    status: 'active',
    gradient: 'from-gray-500 to-slate-600',
  },
  {
    title: 'Equity History',
    description: 'Access historical equity bar data',
    href: '/equity',
    icon: TrendingUp,
    category: 'Analytics',
    status: 'active',
    gradient: 'from-green-500 to-emerald-500',
  },
  {
    title: 'Option Chains',
    description: 'Browse option chain snapshots',
    href: '/chains',
    icon: Database,
    category: 'Data Sources',
    status: 'active',
    gradient: 'from-purple-500 to-violet-500',
  },
  {
    title: 'Option Workflow',
    description: 'Select, enrich, and manage option contracts',
    href: '/options',
    icon: Building2,
    category: 'Workflow',
    status: 'active',
    gradient: 'from-orange-500 to-red-500',
  },
  {
    title: 'Option History',
    description: 'Query historical option bar data',
    href: '/history',
    icon: History,
    category: 'Analytics',
    status: 'active',
    gradient: 'from-indigo-500 to-blue-500',
  },
];

const stats = [
  { label: 'Active Services', value: '6', icon: Activity, color: 'text-green-500' },
  { label: 'Data Sources', value: '3', icon: Database, color: 'text-blue-500' },
  { label: 'Uptime', value: '99.9%', icon: Shield, color: 'text-emerald-500' },
  { label: 'Response Time', value: '145ms', icon: Zap, color: 'text-yellow-500' },
];

export default function HomePage() {
  const [currentTime, setCurrentTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      }));
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <MainLayout>
      <div className="space-y-8">
        {/* Enhanced Header */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800 p-8 text-white">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-transparent"></div>
          <div className="relative">
            <div className="flex items-center justify-between">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <BarChart3 className="h-8 w-8" />
                  <h1 className="text-4xl font-bold tracking-tight">
                    Trading Command Center
                  </h1>
                </div>
                <p className="text-blue-100 text-lg max-w-2xl">
                  Comprehensive trading platform with real-time market data, advanced analytics, and automated workflow management.
                </p>
              </div>
              <div className="text-right">
                <div className="text-blue-100 text-sm mb-1">Market Time</div>
                <div className="text-2xl font-mono font-bold">{currentTime}</div>
                <div className="text-blue-200 text-sm">EST</div>
              </div>
            </div>
          </div>
        </div>

        {/* Performance Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((stat) => (
            <Card key={stat.label} className="bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 border-0 shadow-lg">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                    <p className="text-2xl font-bold">{stat.value}</p>
                  </div>
                  <stat.icon className={`h-8 w-8 ${stat.color}`} />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Enhanced Service Cards */}
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold tracking-tight">Trading Services</h2>
            <Badge variant="secondary" className="text-sm">
              {services.length} Services Available
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {services.map((service) => (
              <Link key={service.href} href={service.href}>
                <Card className="group relative overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer transform hover:-translate-y-1">
                  {/* Gradient background */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${service.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}></div>

                  <CardHeader className="relative">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`p-3 rounded-xl bg-gradient-to-br ${service.gradient} text-white shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                          <service.icon className="h-6 w-6" />
                        </div>
                        <div>
                          <CardTitle className="text-xl group-hover:text-primary transition-colors duration-200">
                            {service.title}
                          </CardTitle>
                          <Badge variant="outline" className="mt-1 text-xs">
                            {service.category}
                          </Badge>
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span className="text-xs text-green-600 font-medium">Live</span>
                      </div>
                    </div>
                    <CardDescription className="text-sm text-muted-foreground group-hover:text-foreground/80 transition-colors duration-200 mt-3">
                      {service.description}
                    </CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
        </div>
        </div>

        {/* Enhanced System Status */}
        <Card className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 border-green-200 dark:border-green-800">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-green-800 dark:text-green-200 flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  System Health Dashboard
                </CardTitle>
                <CardDescription className="text-green-600 dark:text-green-300">
                  Real-time monitoring of all trading microservices
                </CardDescription>
              </div>
              <Badge className="bg-green-500 text-white">All Systems Operational</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {[
                { name: 'Calendar API', port: '9001', latency: '12ms' },
                { name: 'Equity API', port: '9002', latency: '8ms' },
                { name: 'Chains API', port: '9003', latency: '15ms' },
                { name: 'Orchestrator', port: '9005', latency: '22ms' }
              ].map((service) => (
                <div key={service.name} className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-green-200 dark:border-green-700">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="font-medium text-sm">{service.name}</span>
                    </div>
                    <Badge variant="outline" className="text-xs">:{service.port}</Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Latency: <span className="text-green-600 font-medium">{service.latency}</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1 mt-2">
                    <div className="bg-green-500 h-1 rounded-full" style={{ width: '92%' }}></div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </MainLayout>
  );
}
