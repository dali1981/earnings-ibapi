import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV || 'development',
  release: process.env.SENTRY_RELEASE || '1.0.0',
  tracesSampleRate: parseFloat(process.env.SENTRY_TRACES_SAMPLE_RATE || '0.1'),

  // Performance monitoring for API routes
  integrations: [
    Sentry.httpIntegration({ tracing: true })
  ],

  beforeSend(event) {
    // Server-side filtering
    if (event.request && event.request.url) {
      // Skip health check endpoints
      if (event.request.url.includes('/health')) {
        return null;
      }
    }

    return event;
  }
});