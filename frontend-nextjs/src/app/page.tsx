import { MainLayout } from '@/components/layout/MainLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import Link from 'next/link';
import { Calendar, TrendingUp, Database, Building2, History, Settings } from 'lucide-react';

const services = [
  {
    title: 'Calendar Service',
    description: 'View and fetch upcoming earnings announcements',
    href: '/calendar',
    icon: Calendar,
  },
  {
    title: 'Contract Management',
    description: 'Search, populate, and setup contract descriptions',
    href: '/contracts',
    icon: Settings,
  },
  {
    title: 'Equity History',
    description: 'Access historical equity bar data',
    href: '/equity', 
    icon: TrendingUp,
  },
  {
    title: 'Option Chains',
    description: 'Browse option chain snapshots',
    href: '/chains',
    icon: Database,
  },
  {
    title: 'Option Workflow',
    description: 'Select, enrich, and manage option contracts',
    href: '/options',
    icon: Building2,
  },
  {
    title: 'Option History',
    description: 'Query historical option bar data',
    href: '/history',
    icon: History,
  },
];

export default function HomePage() {
  return (
    <MainLayout>
      <div className="space-y-8">
        {/* Header */}
        <div className="space-y-4">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Trading Services Dashboard
          </h1>
          <p className="text-muted-foreground text-lg">
            Manage and monitor your trading microservices from a unified interface.
          </p>
        </div>

        {/* Service Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {services.map((service) => (
            <Link key={service.href} href={service.href}>
              <Card className="h-full bg-card border-border hover:bg-accent/50 hover:border-ring/50 hover:shadow-lg transition-all duration-300 cursor-pointer group">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <service.icon className="h-6 w-6 text-primary group-hover:text-ring transition-colors duration-200" />
                    <CardTitle className="text-xl text-card-foreground group-hover:text-accent-foreground transition-colors duration-200">{service.title}</CardTitle>
                  </div>
                  <CardDescription className="text-sm text-muted-foreground group-hover:text-foreground/80 transition-colors duration-200">
                    {service.description}
                  </CardDescription>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>

        {/* Status Section */}
        <Card className="bg-card/50 border-border">
          <CardHeader>
            <CardTitle className="text-card-foreground">System Status</CardTitle>
            <CardDescription className="text-muted-foreground">
              Current status of all trading microservices
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="flex items-center gap-2 text-foreground">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                Calendar API
              </div>
              <div className="flex items-center gap-2 text-foreground">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                Equity API
              </div>
              <div className="flex items-center gap-2 text-foreground">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                Chains API
              </div>
              <div className="flex items-center gap-2 text-foreground">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                Orchestrator API
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </MainLayout>
  );
}
