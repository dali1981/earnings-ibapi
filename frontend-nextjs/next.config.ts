import { withSentryConfig } from '@sentry/nextjs';
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  env: {
    CALENDAR_API_URL: process.env.NEXT_PUBLIC_CALENDAR_API_URL || 'http://localhost:9001',
    EQUITY_API_URL: process.env.NEXT_PUBLIC_EQUITY_API_URL || 'http://localhost:9002',
    CHAINS_API_URL: process.env.NEXT_PUBLIC_CHAINS_API_URL || 'http://localhost:9004',
    ORCHESTRATOR_API_URL: process.env.NEXT_PUBLIC_ORCHESTRATOR_API_URL || 'http://localhost:9005',
  },
  async rewrites() {
    return [
      {
        source: '/api/calendar/:path*',
        destination: `${process.env.NEXT_PUBLIC_CALENDAR_API_URL || 'http://localhost:9001'}/calendar/:path*`,
      },
      {
        source: '/api/equity/:path*',
        destination: `${process.env.NEXT_PUBLIC_EQUITY_API_URL || 'http://localhost:9002'}/equity/:path*`,
      },
      {
        source: '/api/chains/:path*',
        destination: `${process.env.NEXT_PUBLIC_CHAINS_API_URL || 'http://localhost:9004'}/chains/:path*`,
      },
      {
        source: '/api/orchestrator/:path*',
        destination: `${process.env.NEXT_PUBLIC_ORCHESTRATOR_API_URL || 'http://localhost:9005'}/orchestrator/:path*`,
      },
    ]
  },
};

// Sentry Webpack Plugin options
const sentryWebpackPluginOptions = {
  // Sentry Webpack Plugin options
  silent: true,
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  authToken: process.env.SENTRY_AUTH_TOKEN,

  // Upload source maps in production only
  include: ['.next'],
  ignore: ['node_modules'],
  dryRun: process.env.NODE_ENV !== 'production'
};

export default withSentryConfig(nextConfig, sentryWebpackPluginOptions);
