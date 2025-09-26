import { Metadata } from 'next';
import { SentryTestComponent } from '@/components/testing/SentryTestComponent';
import { MainLayout } from '@/components/layout/MainLayout';

export const metadata: Metadata = {
  title: 'Sentry Test - Trading Services',
  description: 'Test Sentry integration and observability features',
};

export default function SentryTestPage() {
  return (
    <MainLayout>
      <div className="container mx-auto p-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Sentry Integration Test</h1>
          <p className="text-muted-foreground mt-2">
            Test various Sentry features to verify the integration is working correctly.
          </p>
        </div>
        <SentryTestComponent />
      </div>
    </MainLayout>
  );
}