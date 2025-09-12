# Component Architecture

## Overview

The frontend follows a modular component architecture with clear separation between layout, business logic, and UI components.

## Directory Structure

```
src/components/
├── layout/                 # Layout and navigation components
│   ├── MainLayout.tsx      # Main app layout wrapper
│   ├── Sidebar.tsx         # Navigation sidebar
│   └── Breadcrumb.tsx      # Navigation breadcrumbs
├── trading/                # Business logic components
│   ├── Calendar/           # Earnings calendar functionality
│   ├── Equity/             # Equity data management
│   ├── Chains/             # Option chains browsing
│   ├── Contracts/          # Option contract workflow
│   ├── History/            # Option history interface
│   └── Orchestrator/       # Contract orchestration
└── ui/                     # shadcn/ui components
    ├── button.tsx          # Button component
    ├── card.tsx            # Card component
    ├── table.tsx           # Table component
    └── ...                 # Other UI primitives
```

## Layout Components

### MainLayout
**Purpose**: Root layout wrapper for all pages
**Features**:
- Responsive sidebar integration
- Breadcrumb navigation
- Content area with proper spacing
- Mobile-friendly design

```typescript
interface MainLayoutProps {
  children: React.ReactNode;
}
```

### Sidebar
**Purpose**: Primary navigation interface
**Features**:
- Collapsible mobile menu
- Active route highlighting
- Home button with special styling
- Responsive design with overlay

**Navigation Structure**:
- Home (`/`) - Dashboard overview
- Contracts (`/contracts`) - Contract management
- Equity History (`/equity`) - Historical equity data
- Option Chains (`/chains`) - Option chain browsing
- Option Workflow (`/options`) - Option contract workflow
- Option History (`/history`) - Historical option data

## Trading Components

### Calendar Component
**Location**: `src/components/trading/Calendar/`
**Purpose**: Earnings calendar management

**Features**:
- Date range filtering
- Symbol-based filtering
- Table/JSON view toggle
- Data fetching with loading states
- Metadata extraction from API responses

**API Integration**:
- `GET /calendar/events` - Fetch earnings events
- `POST /calendar/fetch` - Trigger new data fetch

### Equity Component
**Location**: `src/components/trading/Equity/`
**Purpose**: Historical equity data management

**Features**:
- Tabbed interface (Read Repository / Fetch from TWS)
- Date range selection and validation
- Bar size configuration
- Available dates checking
- Form validation with error handling

**Tabs**:
- **Read Repository**: Query existing data by date ranges
- **Fetch from TWS**: Pull new data via orchestrator

### Option Chains Component
**Location**: `src/components/trading/Chains/`
**Purpose**: Option chain snapshot browsing

**Features**:
- Symbol-based search
- Available dates browsing
- As-of date selection
- JSON data exploration
- Snapshot fetching via orchestrator

### Orchestrator Component
**Location**: `src/components/trading/Orchestrator/`
**Purpose**: Contract search and setup operations

**Features**:
- Multi-field configuration form
- Symbol, security type, currency, and exchange settings
- Three operation modes:
  - **Search Descriptions**: Find contract descriptions
  - **Fetch Details**: Get detailed contract information
  - **Setup**: Combined search + details operation
- Real-time status updates
- JSON output display

### Option Contracts Component
**Location**: `src/components/trading/Contracts/`
**Purpose**: Complete option contract workflow

**Complex Multi-Tab Interface**:
- **Select**: Contract selection with filtering
- **Enrich**: Add detailed contract information
- **Save**: Persist contracts to repository
- **Results**: View contracts in table/JSON format
- **Historical**: Fetch option bar data
- **Preview**: Browse saved option bar data

**Advanced Features**:
- Polling for async task completion
- Contract availability checking
- Expiry date management
- Multi-step workflow state management
- Table rendering with enrichment status

### Option History Component
**Location**: `src/components/trading/History/`
**Purpose**: Historical option data interface

**Implementation**: Wrapper component that opens Option Contracts with Historical tab active
```typescript
export default function OptionHistoryComponent() {
  return <OptionContractsComponent initialActive="historical" />;
}
```

## UI Component System

### shadcn/ui Integration
All UI components are built on **shadcn/ui** primitives for consistency and accessibility.

**Key Components Used**:
- **Card**: Content containers with headers and descriptions
- **Button**: Primary actions with variants (default, outline, ghost)
- **Input**: Form inputs with validation states
- **Select**: Dropdown selections with search
- **Table**: Data display with sorting and selection
- **Tabs**: Multi-panel interfaces
- **Badge**: Status indicators and labels
- **Toast**: User feedback notifications

### Component Patterns

#### Form Handling
```typescript
// Consistent form state management
const [formData, setFormData] = useState(initialState);
const [loading, setLoading] = useState(false);
const [errors, setErrors] = useState<string[]>([]);

// Form submission with error handling
const handleSubmit = async () => {
  setLoading(true);
  setErrors([]);
  try {
    const result = await apiCall(formData);
    toast.success('Operation completed successfully');
  } catch (error) {
    setErrors([error.message]);
    toast.error('Operation failed');
  } finally {
    setLoading(false);
  }
};
```

#### Data Display
```typescript
// Loading states with skeleton placeholders
if (loading) {
  return <Skeleton className="h-4 w-full" />;
}

// Error states with user-friendly messages
if (error) {
  return <div className="text-destructive">Failed to load data</div>;
}

// Data rendering with fallbacks
return (
  <Table>
    {data.map(item => (
      <TableRow key={item.id}>
        <TableCell>{item.value || 'N/A'}</TableCell>
      </TableRow>
    ))}
  </Table>
);
```

#### Async Operations
```typescript
// Polling pattern for long-running operations
useEffect(() => {
  if (taskId && !pollTimerRef.current) {
    pollTimerRef.current = setInterval(async () => {
      try {
        const result = await checkTaskStatus(taskId);
        if (result.completed) {
          clearInterval(pollTimerRef.current!);
          setResult(result);
          toast.success('Task completed');
        }
      } catch (error) {
        clearInterval(pollTimerRef.current!);
        toast.error('Task failed');
      }
    }, 2000);
  }

  return () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
    }
  };
}, [taskId]);
```

## State Management

### React Query for API State
- Automatic caching and background refetching
- Optimistic updates for mutations
- Loading and error state management
- Query invalidation for data freshness

### Local Component State
- Form data and UI state management
- Loading states for user feedback
- Error states for validation and API errors
- Modal and dialog state management

### URL State
- Route parameters for data filtering
- Query parameters for sharing application state
- Navigation state preservation

## Styling Architecture

### Tailwind CSS Utility Classes
- Consistent spacing and sizing
- Responsive design patterns
- Dark theme support
- Component variants

### CSS Custom Properties
- Theme-aware color system
- Dynamic styling based on user preferences
- Consistent typography scale

### Component-Level Styling
```typescript
// Conditional styling with cn utility
<Card className={cn(
  "base-styles",
  isActive && "active-styles",
  isDisabled && "disabled-styles"
)}>
```

## Performance Optimizations

### Code Splitting
- Route-based code splitting with Next.js App Router
- Dynamic imports for heavy components
- Lazy loading of non-critical functionality

### Memoization
```typescript
// Expensive computations
const expensiveValue = useMemo(() => {
  return computeExpensiveValue(props);
}, [props.dependencies]);

// Callback optimization
const handleClick = useCallback(() => {
  onAction(id);
}, [onAction, id]);
```

### Bundle Optimization
- Tree shaking of unused dependencies
- Optimized imports from UI libraries
- Production build optimization with Next.js

## Testing Strategy

### Component Testing
- Unit tests for business logic
- Integration tests for API interactions
- Accessibility testing with jest-axe
- Visual regression testing

### Testing Utilities
```typescript
// Custom render with providers
const renderWithProviders = (component) => {
  return render(
    <QueryProvider>
      <ThemeProvider>
        {component}
      </ThemeProvider>
    </QueryProvider>
  );
};
```

## Best Practices

### Component Design
1. **Single Responsibility**: Each component has one clear purpose
2. **Composition over Inheritance**: Use composition patterns for flexibility
3. **Props Interface**: Well-defined TypeScript interfaces for all props
4. **Error Boundaries**: Graceful error handling at component boundaries
5. **Accessibility**: ARIA labels and keyboard navigation support

### Code Organization
1. **Consistent File Structure**: Logical grouping of related components
2. **Clear Naming**: Descriptive component and function names
3. **TypeScript First**: Comprehensive type coverage
4. **Documentation**: JSDoc comments for complex components
5. **Reusability**: Shared patterns extracted to reusable components