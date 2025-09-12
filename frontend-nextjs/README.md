# Trading Services Frontend

Modern Next.js frontend for trading microservices built with **shadcn/ui** components and **TypeScript**.

## Overview

This frontend provides a unified interface for managing trading microservices, including earnings calendar, contract management, equity data, and option workflows. Migrated from React to Next.js 15.5.2 with App Router architecture.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

**Development Server:** http://localhost:3002 (auto-assigned if 3000 is in use)

## Service Routes

### 🏠 **Home** `/`
- Service dashboard with clickable cards
- System status monitoring
- Navigation to all trading services

### ⚙️ **Contract Management** `/contracts`
**Purpose:** Contract population and setup operations
- **Search Descriptions** - Find and save contract descriptions for symbols
- **Fetch Details** - Get detailed contract information for symbols
- **Setup Operations** - Combined search + details workflow
- Configure symbols, security types, currency, and exchanges
- Async orchestration with real-time status updates

### 📅 **Calendar Service** `/calendar`
**Purpose:** Earnings announcements management
- Browse upcoming earnings events
- Filter by date range and symbols
- Fetch new earnings data from external sources
- Table and JSON view modes

### 📈 **Equity History** `/equity`
**Purpose:** Historical equity bar data
- **Read Repository** - Query existing historical data by date ranges
- **Fetch from TWS** - Pull new data via orchestrator service
- Date availability checking and gap analysis
- Configurable bar sizes (1 min, 1 hour, 1 day, etc.)

### 🗃️ **Option Chains** `/chains`
**Purpose:** Option chain snapshot browsing
- Browse historical option chain snapshots by date
- Search available as-of dates for symbols
- Fetch new snapshots via orchestrator
- JSON data exploration

### 🏢 **Option Workflow** `/options`
**Purpose:** Complete option contract workflow
- **Select** - Multi-criteria contract selection with filtering
- **Enrich** - Add detailed contract information (conId, localSymbol)
- **Save** - Persist contracts to repository
- **Results** - View selected contracts in table/JSON format
- **Historical** - Fetch option bar data for contracts
- **Preview** - Browse saved option bar data with filtering

### 📊 **Option History** `/history`
**Purpose:** Historical option data focused interface
- Same as Option Workflow but opens directly to Historical tab
- Streamlined interface for fetching option bar history

## Architecture

### **Technology Stack**
- **Next.js 15.5.2** - App Router with TypeScript
- **shadcn/ui** - Modern, accessible UI components
- **Tailwind CSS** - Utility-first styling with dark theme support
- **React Query** - API state management and caching
- **Sonner** - Toast notifications
- **Lucide React** - Icon library

### **Key Features**
- 🌙 **Dark theme optimized** with professional color scheme
- 📱 **Responsive design** - Mobile-friendly with collapsible sidebar
- ⚡ **Real-time updates** - Polling for async operations
- 🔄 **API caching** - Intelligent data fetching with React Query
- 🎯 **Type-safe** - Full TypeScript integration
- ♿ **Accessible** - WCAG compliant components

### **API Integration**
- **Environment-aware configuration** (development/production)
- **Centralized API client** with typed responses
- **Error handling** with user-friendly notifications
- **Request/response interceptors** for headers and metadata

## Project Structure

```
src/
├── app/                    # Next.js App Router pages
│   ├── (routes)/          # Page components
│   ├── globals.css        # Global styles
│   └── layout.tsx         # Root layout
├── components/
│   ├── layout/            # Layout components (Sidebar, MainLayout)
│   ├── trading/           # Business logic components
│   └── ui/                # shadcn/ui components
├── lib/
│   ├── api/               # API client and endpoints
│   └── utils.ts           # Utility functions
└── types/
    └── api.ts             # TypeScript type definitions
```

## Development

### **Commands**
```bash
npm run dev          # Development server
npm run build        # Production build
npm run start        # Start production server
npm run lint         # ESLint checking
```

### **Environment Variables**
```bash
# .env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:9000  # API base URL
```

### **Code Quality**
- **ESLint** - Code linting with Next.js config
- **TypeScript** - Strict type checking
- **Prettier** - Code formatting (via IDE integration)

## API Services Integration

### **Backend Services**
- **Calendar API** (Port 9001) - Earnings data
- **Equity API** (Port 9002) - Historical equity bars  
- **Option History API** (Port 9003) - Option bar data
- **Option Chain API** (Port 9004) - Option chain snapshots
- **Orchestrator API** (Port 9005) - Async workflows

### **Request Flow**
1. **Frontend** makes API calls via centralized client
2. **API client** handles environment-aware base URLs
3. **React Query** manages caching and state
4. **Components** display data with loading/error states

## Migration Notes

This frontend was migrated from a React/Vite application to Next.js with the following improvements:

- ✅ **Route structure corrected** - Contract management now properly at `/contracts`
- ✅ **Modern UI components** - Professional shadcn/ui design system
- ✅ **Better state management** - React Query for API data
- ✅ **Improved navigation** - Responsive sidebar with breadcrumbs
- ✅ **Enhanced UX** - Loading states, error handling, toast notifications
- ✅ **Type safety** - Full TypeScript coverage

## Deployment

### **Production Build**
```bash
npm run build
```

### **Docker Support**
```dockerfile
# Dockerfile included for containerized deployment
# Multi-stage build with optimized production image
```

### **Environment Configuration**
- Configure `NEXT_PUBLIC_API_BASE_URL` for your backend services
- Ensure all backend APIs are accessible from frontend host

## Contributing

1. Follow existing code patterns and TypeScript conventions
2. Use shadcn/ui components for consistency
3. Add proper error handling and loading states
4. Update type definitions when adding new API endpoints
5. Test responsive design on mobile devices
