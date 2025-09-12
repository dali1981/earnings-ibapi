'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Home, ChevronRight } from 'lucide-react'

interface BreadcrumbItem {
  title: string
  href: string
  isHome?: boolean
}

const pathMapping: Record<string, string> = {
  '/': 'Home',
  '/calendar': 'Earnings Calendar',
  '/equity': 'Equity History',
  '/chains': 'Option Chains',
  '/contracts': 'Option Contracts',
  '/history': 'Option History',
  '/orchestrator': 'Orchestrator'
}

export function Breadcrumb() {
  const pathname = usePathname()

  const generateBreadcrumbs = (pathname: string): BreadcrumbItem[] => {
    if (pathname === '/') {
      return [{ title: 'Home', href: '/', isHome: true }]
    }

    const items: BreadcrumbItem[] = [
      { title: 'Home', href: '/', isHome: true }
    ]

    const segments = pathname.split('/').filter(Boolean)
    
    segments.forEach((segment, index) => {
      const href = '/' + segments.slice(0, index + 1).join('/')
      const title = pathMapping[href] || segment.charAt(0).toUpperCase() + segment.slice(1)
      
      items.push({
        title,
        href,
        isHome: false
      })
    })

    return items
  }

  const breadcrumbs = generateBreadcrumbs(pathname)

  return (
    <nav aria-label="Breadcrumb" className="flex items-center space-x-1 text-sm text-muted-foreground mb-4">
      {breadcrumbs.map((item, index) => (
        <div key={item.href} className="flex items-center">
          {index > 0 && (
            <ChevronRight className="h-3 w-3 mx-2" />
          )}
          
          {index === breadcrumbs.length - 1 ? (
            <span className="text-foreground font-medium flex items-center gap-1">
              {item.isHome && <Home className="h-3 w-3" />}
              {item.title}
            </span>
          ) : (
            <Link 
              href={item.href}
              className="hover:text-foreground transition-colors flex items-center gap-1"
            >
              {item.isHome && <Home className="h-3 w-3" />}
              {item.title}
            </Link>
          )}
        </div>
      ))}
    </nav>
  )
}