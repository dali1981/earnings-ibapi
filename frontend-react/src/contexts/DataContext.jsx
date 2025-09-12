import React, { createContext, useContext, useState, useEffect } from 'react'

const DataContext = createContext()

// Cache TTL in milliseconds (5 minutes)
const CACHE_TTL = 5 * 60 * 1000

// Helper to create cache key
const createCacheKey = (component, params) => {
  return `${component}_${JSON.stringify(params)}`
}

// Helper to check if cache is still valid
const isCacheValid = (timestamp) => {
  return Date.now() - timestamp < CACHE_TTL
}

// Helper to save to localStorage
const saveToStorage = (key, data) => {
  try {
    const cacheItem = {
      data,
      timestamp: Date.now()
    }
    localStorage.setItem(key, JSON.stringify(cacheItem))
  } catch (error) {
    console.warn('Failed to save to localStorage:', error)
  }
}

// Helper to load from localStorage
const loadFromStorage = (key) => {
  try {
    const item = localStorage.getItem(key)
    if (!item) return null
    
    const cacheItem = JSON.parse(item)
    if (!isCacheValid(cacheItem.timestamp)) {
      localStorage.removeItem(key)
      return null
    }
    
    return cacheItem.data
  } catch (error) {
    console.warn('Failed to load from localStorage:', error)
    return null
  }
}

export function DataProvider({ children }) {
  const [cache, setCache] = useState({})
  const [metadata, setMetadata] = useState({})

  // Load cache from localStorage on mount
  useEffect(() => {
    try {
      const keys = Object.keys(localStorage)
      const initialCache = {}
      const initialMetadata = {}
      
      keys.forEach(key => {
        const data = loadFromStorage(key)
        if (data) {
          initialCache[key] = data
        }
        
        // Load metadata if exists
        const metaKey = `meta_${key}`
        const meta = loadFromStorage(metaKey)
        if (meta) {
          initialMetadata[key] = meta
        }
      })
      
      setCache(initialCache)
      setMetadata(initialMetadata)
    } catch (error) {
      console.warn('Failed to load initial cache:', error)
    }
  }, [])

  // Cache data with metadata
  const cacheData = (component, params, data, meta = {}) => {
    const key = createCacheKey(component, params)
    
    // Save data
    setCache(prev => ({ ...prev, [key]: data }))
    saveToStorage(key, data)
    
    // Save metadata if provided
    if (Object.keys(meta).length > 0) {
      setMetadata(prev => ({ ...prev, [key]: meta }))
      saveToStorage(`meta_${key}`, meta)
    }
  }

  // Get cached data
  const getCachedData = (component, params) => {
    const key = createCacheKey(component, params)
    
    // Check memory cache first
    if (cache[key]) {
      return cache[key]
    }
    
    // Check localStorage
    const data = loadFromStorage(key)
    if (data) {
      setCache(prev => ({ ...prev, [key]: data }))
      return data
    }
    
    return null
  }

  // Get metadata
  const getMetadata = (component, params) => {
    const key = createCacheKey(component, params)
    
    if (metadata[key]) {
      return metadata[key]
    }
    
    const meta = loadFromStorage(`meta_${key}`)
    if (meta) {
      setMetadata(prev => ({ ...prev, [key]: meta }))
      return meta
    }
    
    return { src: '-', col: '-' }
  }

  // Clear cache for a component
  const clearCache = (component, params) => {
    const key = createCacheKey(component, params)
    
    setCache(prev => {
      const newCache = { ...prev }
      delete newCache[key]
      return newCache
    })
    
    setMetadata(prev => {
      const newMeta = { ...prev }
      delete newMeta[key]
      return newMeta
    })
    
    localStorage.removeItem(key)
    localStorage.removeItem(`meta_${key}`)
  }

  // Clear all cache
  const clearAllCache = () => {
    setCache({})
    setMetadata({})
    
    try {
      const keys = Object.keys(localStorage)
      keys.forEach(key => {
        if (key.startsWith('calendar_') || key.startsWith('equity_') || 
            key.startsWith('chains_') || key.startsWith('meta_')) {
          localStorage.removeItem(key)
        }
      })
    } catch (error) {
      console.warn('Failed to clear localStorage:', error)
    }
  }

  const value = {
    cacheData,
    getCachedData,
    getMetadata,
    clearCache,
    clearAllCache,
    cache,
    metadata
  }

  return (
    <DataContext.Provider value={value}>
      {children}
    </DataContext.Provider>
  )
}

export function useDataPersistence() {
  const context = useContext(DataContext)
  if (!context) {
    throw new Error('useDataPersistence must be used within DataProvider')
  }
  return context
}